#!/usr/bin/env python3
"""
Autonomous Progressive Growth AI Training System
TRAINING ONLY VERSION - No server, pure training

Features:
- Continuously fetches data from Wikipedia & 250+ knowledge domains
- Progressive Growth: 10M → 50M → 100M → 200M → 350M → 500M
- Smart checkpointing: Never overwrites, timestamped saves
- Memory optimized: 5GB RAM with intelligent batching
- Multimodal: Image analysis support
"""

import os
import sys
import time
import json
import math
import random
import requests
import hashlib
import threading
import re
import gc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# DirectML GPU support
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print(f"[GPU] DirectML available — {torch_directml.device_count()} device(s) detected")
    for i in range(torch_directml.device_count()):
        print(f"  GPU {i}: {torch_directml.device_name(i)}")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("[Warning] torch-directml not installed — falling back to CPU")
    print("  Install with: python -m pip install torch-directml")

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[Warning] psutil not installed - memory monitoring disabled")

try:
    from PIL import Image
    import io
    import base64
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    print("[Warning] PIL not installed - image features disabled")

# =============================
# CONFIGURATION
# =============================
class ModelConfig:
    # Batch sizes tuned for GPU (DirectML) — will auto-reduce on OOM
    PRESETS = {
        "10M": {"hidden_dim": 512, "num_layers": 8, "num_heads": 8, "ffn_mult": 4, "batch_size": 2, "grad_accum": 8},
        "50M": {"hidden_dim": 768, "num_layers": 12, "num_heads": 12, "ffn_mult": 4, "batch_size": 8, "grad_accum": 2},
        "100M": {"hidden_dim": 1024, "num_layers": 18, "num_heads": 16, "ffn_mult": 4, "batch_size": 4, "grad_accum": 4},
        "200M": {"hidden_dim": 1280, "num_layers": 24, "num_heads": 20, "ffn_mult": 4, "batch_size": 2, "grad_accum": 8},
        "350M": {"hidden_dim": 1536, "num_layers": 28, "num_heads": 24, "ffn_mult": 4, "batch_size": 2, "grad_accum": 12},
        "500M": {"hidden_dim": 2048, "num_layers": 32, "num_heads": 32, "ffn_mult": 4, "batch_size": 1, "grad_accum": 16},
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

        # Memory-optimized batch settings
        self.BATCH_SIZE = preset["batch_size"]
        self.GRAD_ACCUMULATION_STEPS = preset["grad_accum"]
        
        self.LEARNING_RATE = 3e-4
        self.WARMUP_STEPS = 2000
        self.WEIGHT_DECAY = 0.1
        self.GRAD_CLIP = 1.0
        self.BETA1 = 0.9
        self.BETA2 = 0.95
        self.EPS = 1e-8

        self.DATA_FILE = "data.txt"
        self.TOKENIZER_FILE = "tokenizer.json"
        self.CHECKPOINT_DIR = "checkpoints"
        self.GROWTH_LOG_FILE = "growth_log.json"

        self.CHECKPOINT_INTERVAL = 300  # 5 minutes
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 3000
        self.FETCH_BATCH = 10
        self.FETCH_DELAY = 5
        
        # Growth criteria
        self.GROWTH_LOSS_THRESHOLD = 3.0
        self.GROWTH_STABLE_STEPS = 1000

        # Device selection: DirectML (GPU) > CUDA > CPU
        if DIRECTML_AVAILABLE:
            self.DEVICE = torch_directml.device()
            self.DEVICE_TYPE = "directml"
        elif torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            self.DEVICE_TYPE = "cuda"
        else:
            self.DEVICE = torch.device("cpu")
            self.DEVICE_TYPE = "cpu"

        self.SEED = 42
        
        # Memory management — auto-detect, reserve 400MB for OS
        if PSUTIL_AVAILABLE:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.MAX_MEMORY_GB = round(total_gb - 0.3, 1)
        else:
            self.MAX_MEMORY_GB = 5.6
        # Mixed precision only works reliably with CUDA
        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")

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

    def train(self, texts, min_frequency=2):
        print("Training BPE tokenizer...")
        word_counts = {}
        for text in texts:
            for word in text.strip().split():
                word_with_end = ' '.join(list(word)) + ' </w>'
                word_counts[word_with_end] = word_counts.get(word_with_end, 0) + 1

        vocab = set()
        for word in word_counts:
            for char in word.split():
                vocab.add(char)
        for token in self.special_tokens:
            vocab.add(token)

        print(f"Initial vocabulary size: {len(vocab)}")
        num_merges = self.vocab_size - len(vocab)
        print(f"Learning {num_merges} BPE merges...")
        word_items = list(word_counts.items())

        for i in range(num_merges):
            if i % 1000 == 0:
                print(f"  Merge {i}/{num_merges}")
            pair_counts = {}
            for word, count in word_items:
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = i
            merged = ''.join(best_pair)
            vocab.add(merged)
            new_items = []
            for word, count in word_items:
                symbols = word.split()
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
                        new_symbols.append(merged)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                new_items.append((' '.join(new_symbols), count))
            word_items = new_items

        # Build final vocab
        self.vocab = dict(self.special_tokens)
        current_id = len(self.special_tokens)
        for token in sorted(vocab):
            if token not in self.vocab:
                self.vocab[token] = current_id
                current_id += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizer trained: {len(self.vocab)} tokens, {len(self.merges)} merges")

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

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        self.special_tokens = data['special_tokens']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"Tokenizer loaded from {path}: {len(self.vocab)} tokens")

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

        # Precompute RoPE frequencies
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

        # Grouped-query attention
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

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        # Initialize
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

        # Causal mask
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

            # Top-k sampling
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
# DATASET
# =============================
class ConsumingDataset:
    def __init__(self, data_file, tokenizer, max_seq_len):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.consumed_lines = set()
        self.consumed_file = data_file + ".consumed"
        self.lock = threading.Lock()

        if os.path.exists(self.consumed_file):
            with open(self.consumed_file, 'r') as f:
                self.consumed_lines = set(int(line.strip()) for line in f if line.strip())

    def total_lines(self):
        if not os.path.exists(self.data_file):
            return 0
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)

    def get_batch(self, batch_size):
        with self.lock:
            if not os.path.exists(self.data_file):
                return None

            texts = []
            new_consumed = []
            with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if line_num not in self.consumed_lines and line.strip():
                        texts.append(line.strip())
                        new_consumed.append(line_num)
                        if len(texts) >= batch_size:
                            break

            if not texts:
                return None

            # Tokenize
            all_ids = []
            for text in texts:
                ids = self.tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > self.max_seq_len:
                    ids = ids[:self.max_seq_len]
                all_ids.append(ids)

            # Pad
            max_len = max(len(ids) for ids in all_ids)
            input_ids = []
            labels = []
            for ids in all_ids:
                padded = ids + [self.tokenizer.special_tokens['<pad>']] * (max_len - len(ids))
                input_ids.append(padded[:-1])
                label = padded[1:]
                label = [-100 if x == self.tokenizer.special_tokens['<pad>'] else x for x in label]
                labels.append(label)

            self.consumed_lines.update(new_consumed)
            return torch.tensor(input_ids), torch.tensor(labels)

    def add_lines(self, lines):
        with self.lock:
            with open(self.data_file, 'a', encoding='utf-8') as f:
                for line in lines:
                    if line.strip():
                        f.write(line.strip() + '\n')

    def flush_consumed(self):
        with self.lock:
            with open(self.consumed_file, 'w') as f:
                for line_num in sorted(self.consumed_lines):
                    f.write(f"{line_num}\n")

# =============================
# DATA FETCHER (ALL SOURCES)
# =============================
class DataFetcher:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.fetched_urls = set()

        # All Wikipedia category themes (250+)
        self.wiki_categories = [
            "Science", "Technology", "Mathematics", "Physics", "Chemistry", "Biology",
            "Computer_science", "Engineering", "Medicine", "Astronomy", "Geology",
            "History", "Philosophy", "Psychology", "Sociology", "Economics",
            "Politics", "Geography", "Literature", "Art", "Music", "Film",
            "Sports", "Food", "Religion", "Mythology", "Architecture",
            "Animals", "Plants", "Environment", "Climate_change", "Energy",
            "Transportation", "Space_exploration", "Artificial_intelligence",
            "Machine_learning", "Robotics", "Quantum_mechanics", "Relativity",
            "Evolution", "Genetics", "Neuroscience", "Ecology", "Conservation",
            "Ancient_history", "Medieval_history", "Renaissance", "Industrial_Revolution",
            "World_War_I", "World_War_II", "Cold_War", "Modern_history",
            "Ethics", "Logic", "Metaphysics", "Epistemology", "Aesthetics",
            "Political_philosophy", "Social_philosophy", "Philosophy_of_science",
            "Cognitive_science", "Developmental_psychology", "Social_psychology",
            "Clinical_psychology", "Educational_psychology", "Behavioral_economics",
            "Macroeconomics", "Microeconomics", "International_relations",
            "Comparative_politics", "Political_theory", "Public_policy",
            "Physical_geography", "Human_geography", "Cartography", "Geopolitics",
            "World_literature", "Poetry", "Drama", "Fiction", "Non-fiction",
            "Visual_arts", "Performing_arts", "Contemporary_art", "Art_history",
            "Classical_music", "Jazz", "Rock_music", "Electronic_music", "World_music",
            "Cinema", "Documentary", "Animation", "Film_theory", "Film_history",
            "Olympics", "Football", "Basketball", "Baseball", "Tennis", "Cricket",
            "Cuisine", "Nutrition", "Agriculture", "Gastronomy", "Wine",
            "World_religions", "Theology", "Religious_history", "Spirituality",
            "Greek_mythology", "Roman_mythology", "Norse_mythology", "Egyptian_mythology",
            "Gothic_architecture", "Renaissance_architecture", "Modern_architecture",
            "Mammals", "Birds", "Reptiles", "Amphibians", "Fish", "Insects",
            "Flowering_plants", "Trees", "Fungi", "Algae", "Bacteria",
            "Sustainability", "Renewable_energy", "Pollution", "Biodiversity",
            "Global_warming", "Carbon_footprint", "Environmental_policy",
            "Nuclear_energy", "Solar_power", "Wind_power", "Hydroelectricity",
            "Aviation", "Maritime_transport", "Rail_transport", "Automotive",
            "NASA", "SpaceX", "International_Space_Station", "Mars_exploration",
            "Neural_networks", "Deep_learning", "Natural_language_processing",
            "Computer_vision", "Reinforcement_learning", "Expert_systems",
            "Industrial_robots", "Humanoid_robots", "Autonomous_vehicles",
            "Quantum_computing", "Quantum_entanglement", "Quantum_field_theory",
            "General_relativity", "Special_relativity", "Spacetime", "Black_holes",
            "Natural_selection", "Speciation", "Paleontology", "Evolutionary_biology",
            "DNA", "RNA", "Gene_expression", "Genome", "Genetic_engineering",
            "Brain", "Neurons", "Synapses", "Neuroplasticity", "Consciousness",
            "Ecosystems", "Food_chains", "Biomes", "Habitats", "Symbiosis",
            "Wildlife_conservation", "Endangered_species", "National_parks",
            "Mesopotamia", "Ancient_Egypt", "Ancient_Greece", "Ancient_Rome",
            "Byzantine_Empire", "Feudalism", "Crusades", "Black_Death",
            "Humanism", "Renaissance_art", "Scientific_Revolution",
            "Steam_engine", "Electricity", "Mass_production", "Urbanization",
            "Treaty_of_Versailles", "League_of_Nations", "Trench_warfare",
            "Holocaust", "Atomic_bombings", "United_Nations", "Decolonization",
            "Space_Race", "Vietnam_War", "Berlin_Wall", "Dissolution_of_USSR",
            "Globalization", "Digital_Revolution", "Internet", "Social_media",
            "Utilitarianism", "Deontology", "Virtue_ethics", "Consequentialism",
            "Formal_logic", "Modal_logic", "Predicate_logic", "Set_theory",
            "Ontology", "Cosmology", "Free_will", "Mind-body_problem",
            "Empiricism", "Rationalism", "Skepticism", "Phenomenology",
            "Beauty", "Sublime", "Art_criticism", "Philosophy_of_art",
            "Democracy", "Liberalism", "Conservatism", "Socialism", "Anarchism",
            "Social_contract", "Justice", "Rights", "Freedom", "Equality",
            "Scientific_method", "Falsifiability", "Paradigm_shift", "Reductionism"
        ]

        # Gutenberg top books
        self.gutenberg_ids = list(range(1, 100))

        # Wikipedia vital articles
        self.vital_levels = [1, 2, 3, 4, 5]

        # Search queries for diverse content
        self.search_queries = [
            "fundamental physics concepts", "cellular biology mechanisms",
            "historical civilizations development", "modern technology advances",
            "mathematical theorems proofs", "climate science research",
            "economic theories models", "psychological phenomena studies",
            "philosophical arguments debates", "linguistic structures analysis",
            "chemical reactions processes", "geological formations earth",
            "astronomical observations discoveries", "medical treatments procedures",
            "engineering design principles", "computer algorithms data",
            "political systems governance", "cultural traditions practices",
            "literary movements authors", "artistic techniques styles",
            "musical theory composition", "scientific experiments findings",
            "biological evolution adaptation", "environmental conservation efforts",
            "technological innovations breakthroughs", "social structures dynamics",
            "educational methodologies learning", "agricultural practices farming",
            "architectural design construction", "transportation systems networks",
            "energy sources sustainability", "space exploration missions",
            "quantum mechanics principles", "artificial intelligence applications",
            "robotics automation industry", "neuroscience brain functions",
            "genetics inheritance traits", "ecology ecosystems balance",
            "mythology ancient cultures", "religious beliefs practices",
            "archaeological excavations findings", "anthropological studies societies"
        ]

    def fetch_wikipedia_random(self, count=10):
        lines = []
        for _ in range(count):
            try:
                url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get('extract', '')
                    if text and len(text) > 50:
                        lines.append(text)
                time.sleep(0.5)
            except:
                pass
        return lines

    def fetch_wikipedia_category(self):
        category = random.choice(self.wiki_categories)
        lines = []
        try:
            url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': 20,
                'format': 'json'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                members = data.get('query', {}).get('categorymembers', [])
                for member in members[:10]:
                    title = member.get('title', '')
                    page_lines = self.fetch_wikipedia_page(title)
                    lines.extend(page_lines)
                    time.sleep(0.5)
        except:
            pass
        return lines, category

    def fetch_wikipedia_page(self, title):
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get('extract', '')
                if text and len(text) > 50:
                    return [text]
        except:
            pass
        return []

    def fetch_wikipedia_vital(self):
        level = random.choice(self.vital_levels)
        lines = []
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:Vital_articles_level_{level}',
                'cmlimit': 20,
                'format': 'json'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                members = data.get('query', {}).get('categorymembers', [])
                for member in members[:10]:
                    title = member.get('title', '')
                    page_lines = self.fetch_wikipedia_page(title)
                    lines.extend(page_lines)
                    time.sleep(0.5)
        except:
            pass
        return lines, level

    def fetch_wikipedia_search(self):
        query = random.choice(self.search_queries)
        lines = []
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': 10,
                'format': 'json'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('query', {}).get('search', [])
                for result in results[:5]:
                    title = result.get('title', '')
                    page_lines = self.fetch_wikipedia_page(title)
                    lines.extend(page_lines)
                    time.sleep(0.5)
        except:
            pass
        return lines, query

    def fetch_gutenberg(self):
        book_id = random.choice(self.gutenberg_ids)
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

        for try_url in [url, alt_url]:
            if try_url in self.fetched_urls:
                continue
            try:
                resp = requests.get(try_url, timeout=15)
                if resp.status_code == 200:
                    text = resp.text
                    # Skip headers/footers
                    lines = text.split('\n')
                    start_idx = 0
                    end_idx = len(lines)
                    for i, line in enumerate(lines):
                        if '*** START' in line.upper():
                            start_idx = i + 1
                            break
                    for i in range(len(lines)-1, -1, -1):
                        if '*** END' in lines[i].upper():
                            end_idx = i
                            break
                    content_lines = lines[start_idx:end_idx]
                    content_lines = [l.strip() for l in content_lines if l.strip() and len(l.strip()) > 20]
                    self.fetched_urls.add(try_url)
                    return content_lines[:100]
            except:
                pass
        return []

    def fetch_wikisource(self):
        lines = []
        try:
            url = "https://en.wikisource.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': 5,
                'format': 'json'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get('query', {}).get('random', [])
                for page in pages:
                    page_id = page.get('id')
                    page_url = f"https://en.wikisource.org/w/api.php"
                    page_params = {
                        'action': 'query',
                        'pageids': page_id,
                        'prop': 'extracts',
                        'explaintext': True,
                        'format': 'json'
                    }
                    page_resp = requests.get(page_url, params=page_params, timeout=10)
                    if page_resp.status_code == 200:
                        page_data = page_resp.json()
                        extract = page_data.get('query', {}).get('pages', {}).get(str(page_id), {}).get('extract', '')
                        if extract:
                            page_lines = extract.split('\n')
                            page_lines = [l.strip() for l in page_lines if l.strip() and len(l.strip()) > 20]
                            lines.extend(page_lines[:50])
                    time.sleep(0.5)
        except:
            pass
        return lines, "random"

    def fetch_wikiquote(self):
        lines = []
        try:
            url = "https://en.wikiquote.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': 5,
                'format': 'json'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get('query', {}).get('random', [])
                for page in pages:
                    page_id = page.get('id')
                    page_url = f"https://en.wikiquote.org/w/api.php"
                    page_params = {
                        'action': 'query',
                        'pageids': page_id,
                        'prop': 'extracts',
                        'explaintext': True,
                        'format': 'json'
                    }
                    page_resp = requests.get(page_url, params=page_params, timeout=10)
                    if page_resp.status_code == 200:
                        page_data = page_resp.json()
                        extract = page_data.get('query', {}).get('pages', {}).get(str(page_id), {}).get('extract', '')
                        if extract:
                            page_lines = extract.split('\n')
                            page_lines = [l.strip() for l in page_lines if l.strip() and len(l.strip()) > 10]
                            lines.extend(page_lines[:30])
                    time.sleep(0.5)
        except:
            pass
        return lines

    def fetch_open_textbook(self):
        lines = []
        try:
            # OpenStax API
            url = "https://openstax.org/api/v2/pages/"
            params = {'type': 'books.Book', 'limit': 50}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                books = data.get('items', [])
                if books:
                    book = random.choice(books)
                    book_title = book.get('title', 'Unknown')
                    lines.append(f"Textbook: {book_title}")
                    description = book.get('description', '')
                    if description:
                        lines.extend(description.split('. ')[:10])
        except:
            pass
        return lines

    def run_forever(self):
        print("[DataFetcher] Starting continuous data fetching...")
        sources = [
            ("Wikipedia (random)", lambda: self.fetch_wikipedia_random(self.config.FETCH_BATCH)),
            ("Wikipedia (category)", lambda: self.fetch_wikipedia_category()[0]),
            ("Wikipedia (vital)", lambda: self.fetch_wikipedia_vital()[0]),
            ("Wikipedia (search)", lambda: self.fetch_wikipedia_search()[0]),
            ("Project Gutenberg", lambda: self.fetch_gutenberg()),
            ("Wikisource", lambda: self.fetch_wikisource()[0]),
            ("Wikiquote", lambda: self.fetch_wikiquote()),
            ("Open Textbook", lambda: self.fetch_open_textbook()),
        ]

        cycle_count = 0
        while self.running:
            cycle_count += 1
            src_name, src_fn = sources[cycle_count % len(sources)]
            
            try:
                lines = src_fn()
                if lines:
                    self.dataset.add_lines(lines)
                    total = self.dataset.total_lines()
                    print(f"[DataFetcher] {src_name}: +{len(lines)} lines (total: {total})")
            except Exception as e:
                print(f"[DataFetcher] {src_name} error: {str(e)[:100]}")
            
            time.sleep(self.config.FETCH_DELAY)

# =============================
# MODEL GROWTH MANAGER
# =============================
class ModelGrowthManager:
    def __init__(self, config):
        self.config = config
        self.growth_log_file = config.GROWTH_LOG_FILE
        self.growth_log = self.load_growth_log()
        self.current_size_idx = ModelConfig.GROWTH_PATH.index(config.SIZE)
        
    def load_growth_log(self):
        if os.path.exists(self.growth_log_file):
            with open(self.growth_log_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_growth_log(self):
        with open(self.growth_log_file, 'w') as f:
            json.dump(self.growth_log, f, indent=2)
    
    def should_grow(self, recent_losses):
        if self.current_size_idx >= len(ModelConfig.GROWTH_PATH) - 1:
            return False
        
        if len(recent_losses) < self.config.GROWTH_STABLE_STEPS:
            return False
        
        avg_loss = sum(recent_losses) / len(recent_losses)
        return avg_loss < self.config.GROWTH_LOSS_THRESHOLD
    
    def grow_model(self, current_model, current_step, avg_loss):
        next_size = ModelConfig.GROWTH_PATH[self.current_size_idx + 1]
        
        print(f"\n{'='*60}")
        print(f"  GROWING MODEL: {self.config.SIZE} → {next_size}")
        print(f"  Current step: {current_step}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Save growth event
        growth_event = {
            "from_size": self.config.SIZE,
            "to_size": next_size,
            "step": current_step,
            "avg_loss": avg_loss,
            "timestamp": datetime.now().isoformat()
        }
        self.growth_log.append(growth_event)
        self.save_growth_log()
        
        # Save final checkpoint of current size
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f"final_{self.config.SIZE}_step{current_step}_{timestamp}.pt"
        )
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        
        checkpoint = {
            'model_state': current_model.state_dict(),
            'step': current_step,
            'loss': avg_loss,
            'size': self.config.SIZE,
            'timestamp': timestamp
        }
        torch.save(checkpoint, final_path)
        print(f"Saved final checkpoint: {final_path}")
        
        # Create new config and model
        new_config = ModelConfig(next_size)
        new_config.DATA_FILE = self.config.DATA_FILE
        new_config.TOKENIZER_FILE = self.config.TOKENIZER_FILE
        new_config.CHECKPOINT_DIR = self.config.CHECKPOINT_DIR
        new_config.VOCAB_SIZE = self.config.VOCAB_SIZE
        
        new_model = AdvancedTransformer(new_config).to(new_config.DEVICE)
        
        # Transfer compatible weights
        print("Transferring compatible weights...")
        old_state = current_model.state_dict()
        new_state = new_model.state_dict()
        
        transferred = 0
        for name, param in old_state.items():
            if name in new_state:
                new_param = new_state[name]
                if param.shape == new_param.shape:
                    new_state[name] = param
                    transferred += 1
        
        new_model.load_state_dict(new_state)
        print(f"Transferred {transferred} compatible parameter tensors")
        
        self.current_size_idx += 1
        self.config = new_config
        
        return new_model, new_config

# =============================
# TRAINER
# =============================
class ContinuousTrainer:
    def __init__(self, model, tokenizer, config, growth_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.growth_manager = growth_manager
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY
        )
        self.step = 0
        self.running = True
        self.paused = False
        self.scaler = torch.cuda.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
        self.recent_losses = []
        self.last_checkpoint_time = time.time()

    def get_lr(self):
        if self.step < self.config.WARMUP_STEPS:
            return self.config.LEARNING_RATE * (self.step + 1) / self.config.WARMUP_STEPS
        return self.config.LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * self.step / 100000))

    def load_checkpoint(self):
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, f"latest_{self.config.SIZE}.pt")
        
        if os.path.exists(latest_path):
            try:
                checkpoint = torch.load(latest_path, map_location=self.config.DEVICE)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.step = checkpoint['step']
                if self.scaler and 'scaler_state' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state'])
                print(f"Resumed from checkpoint: step {self.step}, loss {checkpoint.get('loss', 'N/A')}")
            except Exception as e:
                print(f"Could not load checkpoint: {e}")

    def save_checkpoint(self):
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        
        # Timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f"backup_{self.config.SIZE}_step{self.step}_{timestamp}.pt"
        )
        
        # Latest checkpoint
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, f"latest_{self.config.SIZE}.pt")
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step': self.step,
            'loss': self.recent_losses[-1] if self.recent_losses else None,
            'timestamp': timestamp,
            'size': self.config.SIZE
        }
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()
        
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, backup_path)
        print(f"[Checkpoint] Saved: {latest_path} + {backup_path}")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def train_forever(self, dataset):
        print(f"[Trainer] Starting training on {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                batch = dataset.get_batch(self.config.BATCH_SIZE)
                if batch is None:
                    time.sleep(1)
                    continue

                input_ids, labels = batch
                input_ids = input_ids.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)

                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.get_lr()

                # Forward with mixed precision if enabled
                if self.config.ENABLE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        _, loss = self.model(input_ids, labels)
                    loss = loss / self.config.GRAD_ACCUMULATION_STEPS
                    self.scaler.scale(loss).backward()
                else:
                    _, loss = self.model(input_ids, labels)
                    loss = loss / self.config.GRAD_ACCUMULATION_STEPS
                    loss.backward()

                if (self.step + 1) % self.config.GRAD_ACCUMULATION_STEPS == 0:
                    if self.config.ENABLE_MIXED_PRECISION:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()

                self.step += 1
                current_loss = loss.item() * self.config.GRAD_ACCUMULATION_STEPS
                self.recent_losses.append(current_loss)
                
                # Keep only recent losses for growth decision
                if len(self.recent_losses) > self.config.GROWTH_STABLE_STEPS:
                    self.recent_losses.pop(0)

                # Logging
                if self.step % 10 == 0:
                    avg_loss = sum(self.recent_losses[-100:]) / min(len(self.recent_losses), 100)
                    lr = self.optimizer.param_groups[0]['lr']
                    mem_used = "N/A"
                    if PSUTIL_AVAILABLE:
                        mem_used = f"{psutil.virtual_memory().used / (1024**3):.1f}GB"
                    print(f"[Train] Step {self.step} | Size: {self.config.SIZE} | Loss: {current_loss:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e} | Mem: {mem_used}")

                # Periodic checkpoint
                if time.time() - self.last_checkpoint_time > self.config.CHECKPOINT_INTERVAL:
                    self.save_checkpoint()
                    dataset.flush_consumed()
                    self.last_checkpoint_time = time.time()

                # Check for growth
                if self.growth_manager.should_grow(self.recent_losses):
                    avg_loss = sum(self.recent_losses) / len(self.recent_losses)
                    self.save_checkpoint()
                    dataset.flush_consumed()
                    
                    new_model, new_config = self.growth_manager.grow_model(
                        self.model, self.step, avg_loss
                    )
                    
                    # Update trainer with new model/config
                    self.model = new_model
                    self.config = new_config
                    self.optimizer = torch.optim.AdamW(
                        new_model.parameters(),
                        lr=new_config.LEARNING_RATE,
                        betas=(new_config.BETA1, new_config.BETA2),
                        eps=new_config.EPS,
                        weight_decay=new_config.WEIGHT_DECAY
                    )
                    self.scaler = torch.cuda.amp.GradScaler() if new_config.ENABLE_MIXED_PRECISION else None
                    self.recent_losses = []
                    self.step = 0
                    
                    print(f"Growth complete! Now training {new_config.SIZE} model")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] Reducing batch size: {self.config.BATCH_SIZE} → {self.config.BATCH_SIZE // 2}")
                    self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(5)
                else:
                    print(f"[Error] {str(e)[:200]}")
                    time.sleep(1)
            except Exception as e:
                print(f"[Error] {str(e)[:200]}")
                time.sleep(1)

# =============================
# MAIN APPLICATION
# =============================
class AIApplication:
    def __init__(self, model_size="10M"):
        self.config = ModelConfig(model_size)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.fetcher = None
        self.dataset = None
        self.growth_manager = ModelGrowthManager(self.config)

        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
        if DIRECTML_AVAILABLE:
            torch.manual_seed(self.config.SEED)

        print(f"\n{'='*60}")
        print(f"  AUTONOMOUS PROGRESSIVE GROWTH AI - TRAINING ONLY")
        print(f"{'='*60}")
        print(f"  Starting size : {model_size}")
        print(f"  Growth path   : {' → '.join(ModelConfig.GROWTH_PATH)}")
        gpu_name = torch_directml.device_name(0) if DIRECTML_AVAILABLE else "N/A"
        print(f"  Device        : {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        print(f"  GPU           : {gpu_name}")
        print(f"  Hidden dim    : {self.config.HIDDEN_DIM}")
        print(f"  Layers        : {self.config.NUM_LAYERS}")
        print(f"  Heads         : {self.config.NUM_HEADS}")
        print(f"  Batch size    : {self.config.BATCH_SIZE}")
        print(f"  Grad accum    : {self.config.GRAD_ACCUMULATION_STEPS}")
        print(f"  Max memory    : {self.config.MAX_MEMORY_GB}GB")
        print(f"  Growth loss   : < {self.config.GROWTH_LOSS_THRESHOLD} for {self.config.GROWTH_STABLE_STEPS} steps")
        print(f"  Checkpoint    : every {self.config.CHECKPOINT_INTERVAL}s")
        print(f"  Data sources  : Wikipedia (250+ categories), Gutenberg, Wikisource, Wikiquote")
        print(f"{'='*60}\n")

    def initialize_tokenizer(self):
        if os.path.exists(self.config.TOKENIZER_FILE):
            tok = BPETokenizer()
            tok.load(self.config.TOKENIZER_FILE)
            if len(tok.vocab) >= 1000:
                self.tokenizer = tok
                self.config.VOCAB_SIZE = len(tok.vocab)
                print(f"Tokenizer ready: {len(tok.vocab)} tokens")
                return
            print("Existing tokenizer too small, will retrain...")

        print("Bootstrapping data for tokenizer training...")
        if not os.path.exists(self.config.DATA_FILE):
            with open(self.config.DATA_FILE, 'w') as f:
                pass

        temp_dataset = ConsumingDataset(self.config.DATA_FILE, None, self.config.MAX_SEQ_LEN)
        temp_fetcher = DataFetcher(temp_dataset, self.config)

        bootstrap_sources = [
            ("Wikipedia (random)", lambda: temp_fetcher.fetch_wikipedia_random(20)),
            ("Project Gutenberg", lambda: temp_fetcher.fetch_gutenberg()),
            ("Wikipedia (category)", lambda: temp_fetcher.fetch_wikipedia_category()[0]),
            ("Wikipedia (vital)", lambda: temp_fetcher.fetch_wikipedia_vital()[0]),
            ("Wikipedia (search)", lambda: temp_fetcher.fetch_wikipedia_search()[0]),
            ("Wikisource", lambda: temp_fetcher.fetch_wikisource()[0]),
            ("Wikiquote", lambda: temp_fetcher.fetch_wikiquote()),
            ("Open textbook", lambda: temp_fetcher.fetch_open_textbook()),
        ]
        src_idx = 0
        while temp_dataset.total_lines() < self.config.MIN_DATA_LINES:
            current = temp_dataset.total_lines()
            src_name, src_fn = bootstrap_sources[src_idx % len(bootstrap_sources)]
            print(f"  Data: {current} / {self.config.MIN_DATA_LINES} | Fetching from {src_name}...")
            try:
                lines = src_fn()
                if lines:
                    temp_dataset.add_lines(lines)
            except Exception as e:
                print(f"    {src_name} error: {e}")
            src_idx += 1
            time.sleep(1)

        print(f"Got {temp_dataset.total_lines()} lines. Training tokenizer...")
        texts = []
        with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                if line.strip():
                    texts.append(line.strip())

        self.tokenizer = BPETokenizer(vocab_size=self.config.VOCAB_SIZE)
        self.tokenizer.train(texts)
        self.tokenizer.save(self.config.TOKENIZER_FILE)
        self.config.VOCAB_SIZE = len(self.tokenizer.vocab)
        self.fetched_urls_bootstrap = temp_fetcher.fetched_urls

    def initialize_model(self):
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, f"latest_{self.config.SIZE}.pt")
        
        if os.path.exists(latest_path):
            print(f"Found checkpoint for {self.config.SIZE}, will load in trainer")
        else:
            print(f"No checkpoint for {self.config.SIZE}, creating new model")
        
        self.model = AdvancedTransformer(self.config).to(self.config.DEVICE)

    def run(self):
        # 1. Tokenizer
        self.initialize_tokenizer()

        # 2. Model
        self.initialize_model()

        # 3. Dataset
        self.dataset = ConsumingDataset(
            self.config.DATA_FILE, self.tokenizer, self.config.MAX_SEQ_LEN
        )

        # 4. Data fetcher
        self.fetcher = DataFetcher(self.dataset, self.config)
        if hasattr(self, 'fetched_urls_bootstrap'):
            self.fetcher.fetched_urls.update(self.fetched_urls_bootstrap)
        fetcher_thread = threading.Thread(target=self.fetcher.run_forever, daemon=True)
        fetcher_thread.start()

        # 5. Trainer
        self.trainer = ContinuousTrainer(self.model, self.tokenizer, self.config, self.growth_manager)
        self.trainer.load_checkpoint()
        
        print("\n" + "="*60)
        print("  TRAINING STARTED")
        print("="*60)
        print("  Press Ctrl+C to stop and save checkpoint")
        print("="*60 + "\n")
        
        try:
            self.trainer.train_forever(self.dataset)
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
            self.trainer.running = False
            self.fetcher.running = False
            time.sleep(2)
            self.trainer.save_checkpoint()
            self.dataset.flush_consumed()
            print("Checkpoint saved. You can resume anytime with: python train.py")
            print("Goodbye!")

# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    # Always start with 10M for progressive growth
    model_size = "10M"
    if len(sys.argv) > 1:
        model_size = sys.argv[1]
    
    valid_sizes = ModelConfig.GROWTH_PATH
    if model_size not in valid_sizes:
        print(f"Invalid size. Must be one of: {valid_sizes}")
        print("Using default: 10M")
        model_size = "10M"
    
    print("\n" + "="*60)
    print("  AUTONOMOUS PROGRESSIVE GROWTH AI TRAINER")
    print("  TRAINING ONLY VERSION")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Starts at 10M, grows automatically to 500M")
    print("  ✓ All original Wikipedia/Gutenberg/Wikisource fetching")
    print("  ✓ All original transformer architecture")
    print("  ✓ Smart checkpointing (never overwrites)")
    print("  ✓ 5GB RAM optimized batching")
    print("  ✓ Growth logging with complete history")
    print("\nTo train: python train.py")
    print("To deploy: python serve.py (after training)")
    print("="*60 + "\n")
    
    time.sleep(2)
    
    app = AIApplication(model_size)
    app.run()
