#!/usr/bin/env python3
"""
AI Chat Server - Deployment Only
Load trained model and serve via web interface

Usage: python serve.py [model_size]
Example: python serve.py 50M

The server will:
1. Load the latest checkpoint for the specified model size
2. Start a web server on http://localhost:8080
3. Provide a chat interface to interact with the model
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import warnings
warnings.filterwarnings('ignore')

# DirectML GPU support
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print(f"[GPU] DirectML available — {torch_directml.device_count()} device(s) detected")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("[Warning] torch-directml not installed — falling back to CPU")

# =============================
# CONFIGURATION
# =============================
class ModelConfig:
    PRESETS = {
        "10M": {"hidden_dim": 512, "num_layers": 8, "num_heads": 8, "ffn_mult": 4},
        "50M": {"hidden_dim": 768, "num_layers": 12, "num_heads": 12, "ffn_mult": 4},
        "100M": {"hidden_dim": 1024, "num_layers": 18, "num_heads": 16, "ffn_mult": 4},
        "200M": {"hidden_dim": 1280, "num_layers": 24, "num_heads": 20, "ffn_mult": 4},
        "350M": {"hidden_dim": 1536, "num_layers": 28, "num_heads": 24, "ffn_mult": 4},
        "500M": {"hidden_dim": 2048, "num_layers": 32, "num_heads": 32, "ffn_mult": 4},
    }
    
    GROWTH_PATH = ["10M", "50M", "100M", "200M", "350M", "500M"]

    def __init__(self, size="10M"):
        preset = self.PRESETS.get(size, self.PRESETS["10M"])

        self.SIZE = size
        self.HIDDEN_DIM = preset["hidden_dim"]
        self.NUM_LAYERS = preset["num_layers"]
        self.NUM_HEADS = preset["num_heads"]
        self.NUM_KV_HEADS = preset["num_heads"] // 2
        self.FFN_DIM = preset["hidden_dim"] * preset["ffn_mult"]
        self.HEAD_DIM = self.HIDDEN_DIM // self.NUM_HEADS

        self.VOCAB_SIZE = 32000
        self.MAX_SEQ_LEN = 512
        self.DROPOUT = 0.1
        self.BIAS = False
        self.ROPE_THETA = 10000.0

        self.TOKENIZER_FILE = "tokenizer.json"
        self.CHECKPOINT_DIR = "checkpoints"
        self.CHAT_PORT = 8080

        # Device selection
        if DIRECTML_AVAILABLE:
            self.DEVICE = torch_directml.device()
            self.DEVICE_TYPE = "directml"
        elif torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            self.DEVICE_TYPE = "cuda"
        else:
            self.DEVICE = torch.device("cpu")
            self.DEVICE_TYPE = "cpu"

# =============================
# TOKENIZER
# =============================
class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<sep>': 4, '<cls>': 5, '<mask>': 6
        }
        self.vocab = {}
        self.merges = {}
        self.cache = {}
        self.inverse_vocab = {}

    def apply_merges(self, word):
        symbols = word.split()
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            if not pairs:
                break
            bigram = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if bigram not in self.merges:
                break
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text, add_special_tokens=True):
        if text in self.cache:
            return self.cache[text]

        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])

        words = text.strip().split()
        for word in words:
            word_with_end = ' '.join(list(word)) + ' </w>'
            word_tokens = self.apply_merges(word_with_end)
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.special_tokens['<unk>']))

        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])

        self.cache[text] = tokens
        return tokens

    def decode(self, token_ids):
        tokens = [self.inverse_vocab.get(tid, '<unk>') for tid in token_ids]
        text = ''.join(tokens).replace('</w>', ' ').replace('<bos>', '').replace('<eos>', '')
        return text.strip()

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        self.special_tokens = data['special_tokens']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"Tokenizer loaded: {len(self.vocab)} tokens")

# =============================
# MODEL ARCHITECTURE
# =============================
class RoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.NUM_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.hidden_dim = config.HIDDEN_DIM

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.BIAS)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.BIAS)

        self.dropout = nn.Dropout(config.DROPOUT)

        inv_freq = 1.0 / (config.ROPE_THETA ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def apply_rope(self, x, position_ids):
        seq_len = x.shape[1]
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)

        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(seq_len, device=x.device)
        q = self.apply_rope(q, position_ids)
        k = self.apply_rope(k, position_ids)

        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.o_proj(out)
        return out

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)
        self.w2 = nn.Linear(config.FFN_DIM, config.HIDDEN_DIM, bias=config.BIAS)
        self.w3 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.attn = RoPEAttention(config)
        self.ln2 = nn.LayerNorm(config.HIDDEN_DIM)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class AdvancedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(config.HIDDEN_DIM)
        self.lm_head = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)

        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)

        seq_len = input_ids.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50):
        self.eval()
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.config.DEVICE)

        for _ in range(max_length):
            if input_ids.size(1) >= self.config.MAX_SEQ_LEN:
                break

            logits, _ = self(input_ids)
            next_token_logits = logits[0, -1, :] / temperature

            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_token_idx]

            if next_token.item() == tokenizer.special_tokens['<eos>']:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        generated_tokens = input_ids[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text

# =============================
# CHAT SERVER
# =============================
CHAT_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neural AI Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a0f;--surface:#13131a;--surface2:#1a1a24;--surface3:#23232e;--text:#e8e8f0;--text2:#b0b0c0;--text3:#70707a;--border:#2a2a35;--accent:#6366f1;--accent2:#4f46e5;--accent-dim:rgba(99,102,241,0.15);--user-bg:linear-gradient(135deg,#6366f1,#4f46e5);--ai-bg:var(--surface2)}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);overflow:hidden;height:100vh;display:flex;flex-direction:column}
.header{background:var(--surface);padding:16px 24px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border);box-shadow:0 2px 8px rgba(0,0,0,0.4)}
.logo{font-size:28px;filter:drop-shadow(0 0 8px rgba(99,102,241,0.5))}
.header-info{flex:1}.header-info h1{font-size:1.3em;font-weight:700;letter-spacing:-0.02em}.header-info h1 span{background:var(--user-bg);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.subtitle{font-size:0.72em;color:var(--text3);margin-top:2px}
.stats{display:flex;gap:20px;font-size:0.78em}
.stat{display:flex;align-items:center;gap:6px;color:var(--text2)}
.stat b{color:var(--text);font-weight:600}
.dot{width:7px;height:7px;border-radius:50%;animation:pulse 2s infinite}
.dot.green{background:#10b981;box-shadow:0 0 8px #10b981}
.dot.yellow{background:#f59e0b;box-shadow:0 0 8px #f59e0b}
.dot.red{background:#ef4444;box-shadow:0 0 8px #ef4444}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
#chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
#chat::-webkit-scrollbar{width:6px}
#chat::-webkit-scrollbar-track{background:var(--surface)}
#chat::-webkit-scrollbar-thumb{background:var(--surface3);border-radius:3px}
.welcome{text-align:center;margin:auto;max-width:500px;padding:40px 20px}
.welcome .icon{font-size:64px;margin-bottom:16px;filter:drop-shadow(0 0 12px rgba(99,102,241,0.6))}
.welcome h2{font-size:1.6em;margin-bottom:12px;background:linear-gradient(135deg,var(--text),var(--text2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.welcome p{color:var(--text2);line-height:1.6;font-size:0.92em}
.msg-row{display:flex;gap:12px;max-width:900px;margin:0 auto;width:100%;align-items:flex-start}
.msg-row.user{flex-direction:row-reverse}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}
.msg-row.user .avatar{background:var(--user-bg);box-shadow:0 0 12px rgba(99,102,241,0.4)}
.msg-row.ai .avatar{background:var(--surface2);border:1px solid var(--border)}
.bubble{background:var(--surface2);border:1px solid var(--border);border-radius:16px;padding:14px 18px;max-width:75%;word-wrap:break-word;line-height:1.55;font-size:0.92em;position:relative}
.msg-row.user .bubble{background:var(--user-bg);border:none;color:white;box-shadow:0 2px 8px rgba(99,102,241,0.3)}
.bubble .time{font-size:0.7em;color:var(--text3);margin-top:6px;text-align:right}
.typing{display:flex;gap:4px;padding:2px 0}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--text3);animation:bounce 1.2s infinite}
.typing span:nth-child(2){animation-delay:0.2s}
.typing span:nth-child(3){animation-delay:0.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-8px)}}
.input-wrap{padding:16px 24px 20px;background:var(--surface);border-top:1px solid var(--border);position:relative}
.input-wrap::before{content:'';position:absolute;top:-1px;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent-dim),transparent)}
.input-box{display:flex;align-items:flex-end;gap:10px;background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:4px 4px 4px 16px;transition:border-color 0.2s}
.input-box:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-dim)}
#input{flex:1;background:none;border:none;color:var(--text);font-size:0.92em;font-family:'Inter',sans-serif;padding:10px 0;outline:none;resize:none;max-height:120px;line-height:1.5}
#input::placeholder{color:var(--text3)}
#btn{width:40px;height:40px;border-radius:10px;border:none;background:var(--accent);color:white;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s;flex-shrink:0}
#btn:hover{background:var(--accent2);transform:scale(1.05)}
#btn:active{transform:scale(0.95)}
#btn:disabled{background:var(--surface3);color:var(--text3);cursor:not-allowed;transform:none}
#btn svg{width:18px;height:18px}
.input-hint{font-size:0.68em;color:var(--text3);margin-top:8px;text-align:center}
</style></head><body>

<div class="header">
  <div class="logo">🧠</div>
  <div class="header-info">
    <h1>Neural <span>AI</span></h1>
    <div class="subtitle" id="subtitle">AI Chat Interface</div>
  </div>
  <div class="stats" id="stats">
    <div class="stat"><div class="dot green" id="statusDot"></div> <span id="statusText">Ready</span></div>
    <div class="stat">Model <b id="sModel">—</b></div>
    <div class="stat">Step <b id="sStep">—</b></div>
  </div>
</div>

<div id="chat">
  <div class="welcome" id="welcome">
    <div class="icon">🧠</div>
    <h2>Neural AI Chat</h2>
    <p>This is a trained AI model ready to assist you. Ask questions, have conversations, or explore its knowledge!</p>
  </div>
</div>

<div class="input-wrap">
  <div class="input-box">
    <textarea id="input" rows="1" placeholder="Ask me anything..."
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
    <button id="btn" onclick="send()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </div>
  <div class="input-hint">Press Enter to send · Shift+Enter for new line</div>
</div>

<script>
const chat=document.getElementById('chat');
const input=document.getElementById('input');
const btn=document.getElementById('btn');

input.addEventListener('input',()=>{
  input.style.height='auto';
  input.style.height=Math.min(input.scrollHeight,120)+'px';
});

function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;}
function timeStr(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}

function addMsg(role,text){
  const w=document.getElementById('welcome');if(w)w.remove();
  const row=document.createElement('div');
  row.className='msg-row '+role;
  const av=role==='user'?'👤':'🧠';
  const extra=role==='ai'?'<div class="time">'+timeStr()+'</div>':'';
  row.innerHTML='<div class="avatar">'+av+'</div><div class="bubble">'+esc(text)+extra+'</div>';
  chat.appendChild(row);
  chat.scrollTop=chat.scrollHeight;
}

function showTyping(){
  const w=document.getElementById('welcome');if(w)w.remove();
  const row=document.createElement('div');row.className='msg-row ai';row.id='typing';
  row.innerHTML='<div class="avatar">🧠</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  chat.appendChild(row);chat.scrollTop=chat.scrollHeight;
}
function hideTyping(){const t=document.getElementById('typing');if(t)t.remove();}

async function send(){
  const msg=input.value.trim();if(!msg)return;
  input.value='';input.style.height='auto';
  addMsg('user',msg);
  btn.disabled=true;
  document.getElementById('statusDot').className='dot yellow';
  document.getElementById('statusText').textContent='Thinking...';
  showTyping();
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
    const d=await r.json();
    hideTyping();
    addMsg('ai',d.response);
  }catch(e){hideTyping();addMsg('ai','Connection error.');}
  btn.disabled=false;
  document.getElementById('statusDot').className='dot green';
  document.getElementById('statusText').textContent='Ready';
}

// Load model info
fetch('/api/info').then(r=>r.json()).then(d=>{
  document.getElementById('sModel').textContent=d.size;
  document.getElementById('sStep').textContent=Number(d.step || 0).toLocaleString();
  document.getElementById('subtitle').textContent=d.size+' parameters · '+d.layers+' layers';
}).catch(()=>{});
</script></body></html>"""

class ChatServer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.history = []
        self.checkpoint_info = {}

    def start(self):
        model_ref = self.model
        tokenizer_ref = self.tokenizer
        config_ref = self.config
        history_ref = self.history
        info_ref = self.checkpoint_info

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/chat':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(CHAT_HTML.encode('utf-8'))
                
                elif self.path == '/api/info':
                    info = {
                        'size': config_ref.SIZE,
                        'layers': config_ref.NUM_LAYERS,
                        'hidden_dim': config_ref.HIDDEN_DIM,
                        'step': info_ref.get('step', 0),
                        'device': config_ref.DEVICE_TYPE
                    }
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(info).encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/api/chat':
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length)
                    data = json.loads(body)
                    user_msg = data.get('message', '')

                    try:
                        # Build conversation context
                        prompt = ""
                        for turn in history_ref[-3:]:
                            prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"
                        prompt += f"User: {user_msg}\nAI:"
                        
                        response = model_ref.generate(
                            tokenizer_ref, prompt,
                            max_length=150, temperature=0.8, top_k=40
                        )
                        
                        history_ref.append({
                            'user': user_msg,
                            'ai': response,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        response = f"Error generating response: {str(e)[:100]}"

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response': response}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress request logs

        server = HTTPServer(('0.0.0.0', config_ref.CHAT_PORT), Handler)
        print(f"\n{'='*60}")
        print(f"  SERVER STARTED")
        print(f"{'='*60}")
        print(f"  URL: http://localhost:{config_ref.CHAT_PORT}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            print("Goodbye!")

# =============================
# MAIN
# =============================
def detect_saved_size(model_file="model.pt"):
    """Auto-detect model size from existing model.pt."""
    if os.path.exists(model_file):
        try:
            ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
            size = ckpt.get('size', '10M')
            step = ckpt.get('step', 0)
            print(f"[Resume] Found model.pt — Size: {size}, Step: {step:,}")
            return size
        except Exception as e:
            print(f"[Resume] Could not read model.pt: {e}")
    return None

def load_model(model_size):
    print(f"\n{'='*60}")
    print(f"  NEURAL AI CHAT SERVER")
    print(f"{'='*60}\n")
    
    config = ModelConfig(model_size)
    
    # Check for tokenizer
    if not os.path.exists(config.TOKENIZER_FILE):
        print(f"[Error] Tokenizer not found: {config.TOKENIZER_FILE}")
        print("Please run train.py first to create the tokenizer")
        sys.exit(1)
    
    # Load tokenizer
    print(f"Loading tokenizer from {config.TOKENIZER_FILE}...")
    tokenizer = BPETokenizer()
    tokenizer.load(config.TOKENIZER_FILE)
    config.VOCAB_SIZE = len(tokenizer.vocab)
    
    # Check for model.pt
    model_file = "model.pt"
    if not os.path.exists(model_file):
        print(f"[Error] model.pt not found")
        print("Please run train.py or ai.py first to train a model")
        sys.exit(1)
    
    # Create model
    print(f"Creating {model_size} model...")
    print(f"  Device: {config.DEVICE} ({config.DEVICE_TYPE})")
    print(f"  Hidden dim: {config.HIDDEN_DIM}")
    print(f"  Layers: {config.NUM_LAYERS}")
    print(f"  Heads: {config.NUM_HEADS}")
    
    model = AdvancedTransformer(config).to(config.DEVICE)
    
    # Load from model.pt
    print(f"Loading model from {model_file}...")
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    checkpoint_info = {
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss_history', [0])[-1] if checkpoint.get('loss_history') else 'N/A'
    }
    
    print(f"Model loaded successfully!")
    print(f"  Training step: {checkpoint_info['step']}")
    print(f"  Last loss: {checkpoint_info['loss']}")
    
    return model, tokenizer, config, checkpoint_info

if __name__ == "__main__":
    # Auto-detect model size from model.pt
    saved_size = detect_saved_size()
    if saved_size:
        model_size = saved_size
    else:
        model_size = "10M"
        if len(sys.argv) > 1 and sys.argv[1] in ModelConfig.GROWTH_PATH:
            model_size = sys.argv[1]
    
    # Load model
    model, tokenizer, config, checkpoint_info = load_model(model_size)
    
    # Start server
    server = ChatServer(model, tokenizer, config)
    server.checkpoint_info = checkpoint_info
    server.start()
