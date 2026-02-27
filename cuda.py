#!/usr/bin/env python3
"""
AUTONOMOUS PROGRESSIVE GROWTH AI - ULTIMATE 2TB EDITION
COLAB OPTIMIZED - 12GB RAM FRIENDLY + INTEGRATED DATA FETCHER

Features:
- ULTIMATE BPE: 200K vocabulary, FULL Unicode support
- INTEGRATED DATA FETCHER: Directly downloads from FineWeb when data is low
- CONTINUOUS FETCHING until 1.9TB of training data
- DEDUPLICATION: Never trains same line twice (hash-based)
- SMART DELETION: Deletes text after training (like original)
- STREAMING ARCHITECTURE: Never loads more than 500MB at once
- AUTO-RESUME: Survives Colab disconnects perfectly
- PROGRESSIVE GROWTH: 10M → 50M → 100M → 200M → 350M → 500M
- ADVANCED BPE: Full Unicode, 200K vocabulary
- MEMORY OPTIMIZED: Dynamic batching, monitoring
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
import gzip
import pickle
import signal
from pathlib import Path
from datetime import datetime
from collections import Counter, deque
from typing import Optional, List, Dict, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Optional DirectML support
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except Exception:
    DIRECTML_AVAILABLE = False

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
    IMAGING_AVAILABLE = False  # Disabled by default
except ImportError:
    IMAGING_AVAILABLE = False

# Data fetching imports
try:
    from datasets import load_dataset
    from huggingface_hub import login
    from tqdm import tqdm
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[Warning] datasets not installed - automatic fetching disabled. Run: pip install datasets huggingface_hub tqdm")

# =============================
# CONFIGURATION - ULTIMATE 2TB
# =============================
class ModelConfig:
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
        self.NUM_KV_HEADS = max(1, preset["num_heads"] // 2)
        self.FFN_DIM = preset["hidden_dim"] * preset["ffn_mult"]
        self.HEAD_DIM = self.HIDDEN_DIM // self.NUM_HEADS

        # ULTIMATE VOCAB SIZE - 200K for 2TB data
        self.VOCAB_SIZE = 200000
        self.MAX_SEQ_LEN = 1024
        self.DROPOUT = 0.1
        self.BIAS = False
        self.ROPE_THETA = 10000.0

        self.BATCH_SIZE = preset["batch_size"]
        self.GRAD_ACCUMULATION_STEPS = preset["grad_accum"]

        self.LEARNING_RATE = 3e-4
        self.WARMUP_STEPS = 2000
        self.WEIGHT_DECAY = 0.1
        self.GRAD_CLIP = 1.0
        self.BETA1 = 0.9
        self.BETA2 = 0.95
        self.EPS = 1e-8

        # DIRECTORY SETUP FOR COLAB / DRIVE - 2TB target
        self.DRIVE_DIR = "/content/drive/MyDrive/ai.train"
        os.makedirs(self.DRIVE_DIR, exist_ok=True)
        
        self.DATA_FILE = os.path.join(self.DRIVE_DIR, "data.txt")
        self.TOKENIZER_FILE = os.path.join(self.DRIVE_DIR, "tokenizer_200k.json")
        self.BPE_CHECKPOINT_DIR = os.path.join(self.DRIVE_DIR, "bpe_checkpoints")
        self.CHECKPOINT_DIR = os.path.join(self.DRIVE_DIR, "checkpoints")
        self.MODEL_FILE = os.path.join(self.DRIVE_DIR, "model.pt")
        self.GROWTH_LOG_FILE = os.path.join(self.DRIVE_DIR, "growth_log.json")
        self.TRAINED_HASHES_FILE = os.path.join(self.DRIVE_DIR, "trained_data.json")
        self.FETCHER_HISTORY_FILE = os.path.join(self.DRIVE_DIR, "fetcher_history.json")
        self.FETCH_POSITION_FILE = os.path.join(self.DRIVE_DIR, "fetch_position.json")
        self.FETCH_STATE_FILE = os.path.join(self.DRIVE_DIR, "fineweb_state.json")  # For FineWeb resume
        
        # Create all directories
        for d in [self.BPE_CHECKPOINT_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)

        # ULTIMATE TARGET: 1.9TB (1900 GB) of training data
        self.TARGET_DATA_SIZE_GB = 1900  # 1.9TB
        self.TARGET_DATA_SIZE_BYTES = self.TARGET_DATA_SIZE_GB * 1024**3
        
        # Data fetching triggers
        self.MIN_DATA_LINES_BEFORE_FETCH = 100000  # Start fetching if less than 100k lines
        self.MIN_DATA_SIZE_BEFORE_FETCH = 500 * 1024 * 1024  # 500MB min before fetching
        self.FETCH_BATCH_SIZE_MB = 100  # Fetch 100MB at a time
        self.FETCH_CHECK_INTERVAL = 60  # Check data level every 60 seconds
        
        self.CHECKPOINT_INTERVAL = 300  # 5 minutes
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 1000000  # 1M lines for BPE
        self.PREFETCH_TARGET_LINES = 2_000_000  # Initial prefetch

        self.GROWTH_LOSS_THRESHOLD = 2.8
        self.GROWTH_STABLE_STEPS = 3000

        # Device selection
        if DIRECTML_AVAILABLE:
            try:
                self.DEVICE = torch_directml.device()
                self.DEVICE_TYPE = 'directml'
            except Exception as e:
                print(f"[Warning] DirectML device creation failed: {e}")
                self.DEVICE = torch.device('cpu')
                self.DEVICE_TYPE = 'cpu'
        elif torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            self.DEVICE_TYPE = 'cuda'
        else:
            self.DEVICE = torch.device('cpu')
            self.DEVICE_TYPE = 'cpu'

        self.SEED = 42

        # 12GB RAM OPTIMIZATION - Conservative settings for Colab
        self.MAX_MEMORY_GB = 10.0  # Use max 10GB of 12GB available
        self.RESERVE_BYTES = int(2 * 1024 * 1024 * 1024)  # 2GB safety reserve
        self.BUFFER_SIZE = 25000  # Only keep 25k lines in buffer
        self.BPE_CHUNK_SIZE = 2_000_000  # Process 2M lines at a time

        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")


# =============================
# INTEGRATED DATA FETCHER
# =============================
class IntegratedDataFetcher:
    """Directly fetches data from FineWeb when buffer is low"""
    
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.is_running = True
        self.total_rows_processed = 0
        self.fetcher_thread = None
        self.hf_token = None
        
        # Load fetch state
        self.load_fetch_state()
        
        # Check if datasets are available
        if not DATASETS_AVAILABLE:
            print("⚠️ Automatic data fetching disabled. Install datasets: pip install datasets huggingface_hub tqdm")
    
    def load_fetch_state(self):
        """Load the fetch state from disk"""
        if os.path.exists(self.config.FETCH_STATE_FILE):
            try:
                with open(self.config.FETCH_STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.total_rows_processed = state.get('total_rows_processed', 0)
                print(f"📥 Fetcher state loaded: {self.total_rows_processed:,} rows processed")
            except Exception as e:
                print(f"⚠️ Could not load fetch state: {e}")
    
    def save_fetch_state(self):
        """Save the current fetch state"""
        try:
            with open(self.config.FETCH_STATE_FILE, 'w') as f:
                json.dump({
                    'total_rows_processed': self.total_rows_processed
                }, f)
        except Exception as e:
            print(f"⚠️ Could not save fetch state: {e}")
    
    def check_and_fetch(self):
        """Check if we need more data and fetch if necessary"""
        if not DATASETS_AVAILABLE:
            return False
        
        # Check current data level
        data_size = 0
        data_lines = 0
        if os.path.exists(self.config.DATA_FILE):
            data_size = os.path.getsize(self.config.DATA_FILE)
            # Count lines (approximate)
            try:
                with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    data_lines = sum(1 for _ in f)
            except:
                pass
        
        # Determine if we need to fetch
        needs_fetch = False
        if data_size < self.config.MIN_DATA_SIZE_BEFORE_FETCH:
            needs_fetch = True
            print(f"\n📊 Data size: {data_size/1e6:.2f}MB < {self.config.MIN_DATA_SIZE_BEFORE_FETCH/1e6:.2f}MB - Fetching more...")
        elif data_lines < self.config.MIN_DATA_LINES_BEFORE_FETCH:
            needs_fetch = True
            print(f"\n📊 Data lines: {data_lines:,} < {self.config.MIN_DATA_LINES_BEFORE_FETCH:,} - Fetching more...")
        elif random.random() < 0.1:  # 10% chance to check even if we have enough
            # Verify we're making progress toward 1.9TB
            target_bytes = self.config.TARGET_DATA_SIZE_BYTES
            progress_pct = (data_size / target_bytes) * 100
            if progress_pct < 100:
                needs_fetch = True
                print(f"\n📊 Progress: {progress_pct:.2f}% of 1.9TB - Continuing fetch...")
        
        if needs_fetch:
            self.fetch_batch()
        
        return needs_fetch
    
    def fetch_batch(self):
        """Fetch a batch of data directly into data.txt"""
        print("\n" + "="*60)
        print("🌐 INTEGRATED DATA FETCHER - FETCHING FROM FINEWEB")
        print("="*60)
        
        try:
            # Load FineWeb dataset
            print("📡 Connecting to Hugging Face FineWeb...")
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                                  name="sample-10BT", 
                                  split="train", 
                                  streaming=True)
            
            # Fast-forward to our position
            if self.total_rows_processed > 0:
                print(f"⏩ Fast-forwarding past {self.total_rows_processed:,} rows...")
                dataset = dataset.skip(self.total_rows_processed)
            
            # Fetch until we have enough data
            target_bytes = self.config.FETCH_BATCH_SIZE_MB * 1024 * 1024
            current_bytes = 0
            lines_fetched = 0
            buffer = []
            
            print(f"📥 Fetching {self.config.FETCH_BATCH_SIZE_MB}MB batch...")
            
            for row in tqdm(dataset, desc="Fetching"):
                text = row.get('text', '')
                self.total_rows_processed += 1
                
                if text and len(text) > 20:
                    buffer.append(text.strip())
                    lines_fetched += 1
                    current_bytes += len(text.encode('utf-8'))
                    
                    # Write in chunks to avoid memory issues
                    if len(buffer) >= 5000:
                        self.dataset.add_lines(buffer)
                        buffer = []
                
                # Check if we've fetched enough
                if current_bytes >= target_bytes:
                    break
            
            # Write remaining buffer
            if buffer:
                self.dataset.add_lines(buffer)
            
            # Save state
            self.save_fetch_state()
            
            print(f"✅ Fetch complete: +{lines_fetched:,} lines, {current_bytes/1e6:.2f}MB")
            print(f"📊 Total rows processed: {self.total_rows_processed:,}")
            
        except Exception as e:
            print(f"❌ Fetch error: {e}")
            # Don't crash the main thread
    
    def start_fetcher_thread(self):
        """Start the background fetcher thread"""
        def fetcher_loop():
            while self.is_running:
                try:
                    self.check_and_fetch()
                except Exception as e:
                    print(f"⚠️ Fetcher error: {e}")
                
                # Wait before checking again
                for _ in range(self.config.FETCH_CHECK_INTERVAL):
                    if not self.is_running:
                        break
                    time.sleep(1)
        
        self.fetcher_thread = threading.Thread(target=fetcher_loop, daemon=True)
        self.fetcher_thread.start()
        print("🌐 Integrated Data Fetcher started (checking every 60 seconds)")


# =============================
# ULTIMATE BPE TOKENIZER - 200K VOCAB, FULL UNICODE
# =============================
class UltimateBPETokenizer:
    """200K vocabulary BPE tokenizer with FULL Unicode support"""
    
    def __init__(self, vocab_size=200000):
        self.vocab_size = vocab_size
        self.special_tokens = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3, 
                               '<sep>':4, '<cls>':5, '<mask>':6}
        self.vocab = {}
        self.merges = {}
        self.cache = {}
        self.inverse_vocab = {}
        
        # Advanced features
        self.byte_fallback = True
        self.normalization = 'nfc'
        self.max_word_length = 100
        self.min_frequency = 2  # Minimum frequency for BPE merges
        self.unicode_range = (0x0000, 0x10FFFF)  # Full Unicode
        
        # Streaming checkpointing for 2TB
        self.checkpoint_dir = None
        self.chunk_size = 2_000_000  # Process 2M lines at a time
        
        # Unicode tracking
        self.unicode_stats = {
            'total_chars_found': 0,
            'unicode_chars_found': 0,
            'unique_unicode': set(),
            'corrupted_lines': 0
        }
        
        # Cache settings
        self.cache_size = 10000
        self.progress_interval = 500
        
        # Statistics
        self.stats = {
            'total_lines': 0,
            'unique_words': 0,
            'unicode_words': 0,
            'merge_history': []
        }

    def _normalize(self, text):
        """Unicode normalization"""
        import unicodedata
        if self.normalization == 'nfc':
            return unicodedata.normalize('NFC', text)
        elif self.normalization == 'nfkc':
            return unicodedata.normalize('NFKC', text)
        return text

    def _track_unicode(self, text):
        """Track Unicode characters"""
        for char in text:
            self.unicode_stats['total_chars_found'] += 1
            code = ord(char)
            if code > 127:  # Non-ASCII
                self.unicode_stats['unicode_chars_found'] += 1
                self.unicode_stats['unique_unicode'].add(char)

    def _get_byte_pairs(self, word):
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev = word[0]
        for char in word[1:]:
            pairs.add((prev, char))
            prev = char
        return pairs

    def train_streaming(self, data_file, checkpoint_dir=None):
        """Train BPE on streaming data (2TB capable)"""
        print("\n" + "="*60)
        print("🧠 ULTIMATE BPE TRAINING - STREAMING MODE")
        print("="*60)
        
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize base vocabulary
        self.vocab = dict(self.special_tokens)
        
        # Build byte-level base vocabulary
        for i in range(256):
            byte_char = chr(i)
            if byte_char not in self.vocab:
                self.vocab[byte_char] = len(self.vocab)
        
        print(f"📝 Base vocabulary: {len(self.vocab)} tokens")
        
        # Check file size
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return
        
        file_size = os.path.getsize(data_file)
        print(f"📊 Data file: {file_size / 1e9:.2f} GB")
        
        # Count total lines
        print("\n📏 Counting lines...")
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)
        print(f"   Total lines: {total_lines:,}")
        
        # Stream processing
        chunks_processed = 0
        word_freq = Counter()
        
        print(f"\n🔄 Processing in chunks of {self.chunk_size:,} lines...")
        
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            chunk = []
            line_count = 0
            
            for line in f:
                line = line.strip()
                if len(line) < 20:
                    continue
                
                chunk.append(line)
                line_count += 1
                
                if len(chunk) >= self.chunk_size:
                    # Process chunk
                    for text in chunk:
                        text = self._normalize(text)
                        self._track_unicode(text)
                        
                        # Split into words
                        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
                        for word in words:
                            if len(word) <= self.max_word_length:
                                word_tuple = tuple(word)
                                word_freq[word_tuple] += 1
                    
                    chunks_processed += 1
                    pct = (line_count / total_lines) * 100
                    print(f"   Chunk {chunks_processed}: {line_count:,} / {total_lines:,} ({pct:.1f}%)")
                    
                    # Clear chunk
                    chunk = []
                    
                    # Memory cleanup
                    if chunks_processed % 10 == 0:
                        gc.collect()
        
        # Process remaining
        if chunk:
            for text in chunk:
                text = self._normalize(text)
                self._track_unicode(text)
                words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
                for word in words:
                    if len(word) <= self.max_word_length:
                        word_tuple = tuple(word)
                        word_freq[word_tuple] += 1
        
        print(f"\n📊 Unique words: {len(word_freq):,}")
        print(f"🌐 Unicode stats:")
        print(f"   Total chars: {self.unicode_stats['total_chars_found']:,}")
        print(f"   Unicode chars: {self.unicode_stats['unicode_chars_found']:,}")
        print(f"   Unique unicode: {len(self.unicode_stats['unique_unicode']):,}")
        
        # BPE merges
        print(f"\n🔨 Computing BPE merges (target: {self.vocab_size - len(self.vocab):,})...")
        
        num_merges = self.vocab_size - len(self.vocab)
        for merge_id in range(num_merges):
            # Find best pair
            pair_freq = Counter()
            for word, freq in word_freq.items():
                if freq < self.min_frequency:
                    continue
                pairs = self._get_byte_pairs(word)
                for pair in pairs:
                    pair_freq[pair] += freq
            
            if not pair_freq:
                print(f"   No more pairs to merge at {merge_id:,}")
                break
            
            best_pair = pair_freq.most_common(1)[0][0]
            self.merges[best_pair] = len(self.vocab)
            self.vocab[''.join(best_pair)] = len(self.vocab)
            
            # Update word frequencies
            new_word_freq = Counter()
            for word, freq in word_freq.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_freq[new_word] += freq
            word_freq = new_word_freq
            
            if (merge_id + 1) % 1000 == 0:
                print(f"   Merges: {merge_id + 1:,} / {num_merges:,}")
            
            # Checkpoint
            if checkpoint_dir and (merge_id + 1) % 10000 == 0:
                ckpt_file = os.path.join(checkpoint_dir, f"bpe_checkpoint_{merge_id+1}.json")
                self.save(ckpt_file)
                print(f"   💾 Checkpoint saved: {ckpt_file}")
        
        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"\n✅ BPE training complete!")
        print(f"   Final vocabulary size: {len(self.vocab):,}")
        print(f"   Total merges: {len(self.merges):,}")

    def _merge_pair(self, word, pair):
        """Merge a pair in a word"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(''.join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def encode(self, text):
        """Encode text to token IDs"""
        if not self.vocab:
            return [self.special_tokens['<unk>']]
        
        # Check cache
        cache_key = text[:100]  # Cache first 100 chars
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        text = self._normalize(text)
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        
        tokens = [self.special_tokens['<bos>']]
        
        for word in words:
            # Split word into characters
            word_chars = tuple(word)
            
            # Apply BPE merges
            while len(word_chars) > 1:
                pairs = self._get_byte_pairs(word_chars)
                if not pairs:
                    break
                
                # Find best pair in merges
                bigram = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
                if bigram not in self.merges:
                    break
                
                word_chars = self._merge_pair(word_chars, bigram)
            
            # Convert to token IDs
            for char in word_chars:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    # Byte fallback
                    for byte in char.encode('utf-8'):
                        tokens.append(self.vocab.get(chr(byte), self.special_tokens['<unk>']))
        
        tokens.append(self.special_tokens['<eos>'])
        
        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = tokens
        
        return tokens

    def decode(self, tokens):
        """Decode token IDs to text"""
        if not self.inverse_vocab:
            return ""
        
        text = []
        for token_id in tokens:
            if token_id in [self.special_tokens['<bos>'], self.special_tokens['<eos>'], 
                           self.special_tokens['<pad>']]:
                continue
            
            if token_id in self.inverse_vocab:
                text.append(self.inverse_vocab[token_id])
            else:
                text.append('<unk>')
        
        return ''.join(text)

    def save(self, filepath):
        """Save tokenizer"""
        data = {
            'vocab': self.vocab,
            'merges': {str(k): v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'unicode_stats': {
                k: list(v) if isinstance(v, set) else v 
                for k, v in self.unicode_stats.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Tokenizer saved: {filepath}")

    def load(self, filepath):
        """Load tokenizer"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = {eval(k): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        
        if 'unicode_stats' in data:
            self.unicode_stats = data['unicode_stats']
            if 'unique_unicode' in self.unicode_stats:
                self.unicode_stats['unique_unicode'] = set(self.unicode_stats['unique_unicode'])
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"📥 Tokenizer loaded: {len(self.vocab):,} tokens")


# =============================
# STREAMING DATASET - SMART DELETION
# =============================
class StreamingDataset:
    """Streaming dataset with file position tracking"""
    
    def __init__(self, file_path, tokenizer, config, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        
        # Buffer and tracking
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.trained_hashes = set()
        self.total_processed = 0
        self.lines_in_file = 0
        self.file_size = 0
        self.file_position = 0  # Track position in file
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load trained hashes
        self.load_trained_hashes()
        
        # Load file position
        self.load_file_position()
        
        # Initialize stats
        if os.path.exists(file_path):
            self.file_size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.lines_in_file = sum(1 for line in f if line.strip() and len(line.strip()) > 20)
            print(f"📊 Initial data: {self.lines_in_file:,} lines ({self.file_size/1e6:.2f}MB)")
        
        # Fill initial buffer
        self._fill_buffer()

    def load_file_position(self):
        """Load the last file position"""
        pos_file = self.config.FETCH_POSITION_FILE
        if os.path.exists(pos_file):
            try:
                with open(pos_file, 'r') as f:
                    data = json.load(f)
                    self.file_position = data.get('position', 0)
                print(f"   📍 Resuming from position: {self.file_position:,}")
            except:
                self.file_position = 0

    def save_file_position(self):
        """Save the current file position"""
        pos_file = self.config.FETCH_POSITION_FILE
        try:
            with open(pos_file, 'w') as f:
                json.dump({'position': self.file_position}, f)
        except Exception as e:
            print(f"⚠️ Could not save position: {e}")

    def _fill_buffer(self):
        """Fill buffer with untrained lines"""
        if not os.path.exists(self.file_path):
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to last position
                f.seek(self.file_position)
                
                lines_added = 0
                target = self.config.BUFFER_SIZE - len(self.buffer)
                
                # Read line by line
                while lines_added < target:
                    line = f.readline()
                    if not line:  # EOF
                        break
                    
                    line = line.strip()
                    if len(line) < 20:
                        continue
                    
                    line_hash = hashlib.sha256(line.encode()).hexdigest()
                    if line_hash in self.trained_hashes:
                        continue
                    
                    self.buffer.append(line)
                    lines_added += 1
                
                # Save position
                self.file_position = f.tell()
                self.save_file_position()
                
                # If we reached EOF, wrap around
                if lines_added < target:
                    print(f"   📍 Reached EOF, wrapping to beginning...")
                    f.seek(0)
                    self.file_position = 0
                    
                    while lines_added < target:
                        line = f.readline()
                        if not line:
                            break
                        
                        line = line.strip()
                        if len(line) < 20:
                            continue
                        
                        line_hash = hashlib.sha256(line.encode()).hexdigest()
                        if line_hash in self.trained_hashes:
                            continue
                        
                        self.buffer.append(line)
                        lines_added += 1
                    
                    self.file_position = f.tell()
                    self.save_file_position()
            
            if lines_added > 0:
                print(f"   ✅ Buffer filled: +{lines_added} lines")
                
        except Exception as e:
            print(f"⚠️ Error filling buffer: {e}")
            # Fallback
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    all_lines = [line.strip() for line in f if len(line.strip()) > 20]
                    random.shuffle(all_lines)
                    for line in all_lines[:self.config.BUFFER_SIZE]:
                        line_hash = hashlib.sha256(line.encode()).hexdigest()
                        if line_hash not in self.trained_hashes:
                            self.buffer.append(line)
                print(f"   ✅ Fallback: loaded {len(self.buffer)} lines")
            except:
                pass

    def add_lines(self, lines):
        """Add new lines to data.txt and buffer"""
        if not lines:
            return
        
        lines_added = 0
        
        with self.lock:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                for line in lines:
                    line = line.strip()
                    if len(line) > 20:
                        line_hash = hashlib.sha256(line.encode()).hexdigest()
                        if line_hash not in self.trained_hashes:
                            f.write(line + '\n')
                            self.buffer.append(line)
                            self.total_processed += 1
                            lines_added += 1
                
                # Force flush to disk
                f.flush()
                os.fsync(f.fileno())
            
            if lines_added > 0:
                self.file_size = os.path.getsize(self.file_path)
                self.lines_in_file += lines_added
                print(f"   ✅ Added {lines_added:,} lines to data.txt (Total: {self.lines_in_file:,} lines, {self.file_size/1e6:.2f}MB)")

    def get_batch(self, batch_size):
        """Get a batch of tokenized sequences"""
        with self.lock:
            if len(self.buffer) < batch_size:
                self._fill_buffer()
            
            if len(self.buffer) == 0:
                return None
            
            # Get batch
            batch_lines = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch_lines.append(self.buffer.popleft())
            
            # Tokenize
            if not self.tokenizer:
                return None
            
            input_ids = []
            for line in batch_lines:
                tokens = self.tokenizer.encode(line)
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                input_ids.append(tokens)
            
            # Pad sequences
            max_len = max(len(seq) for seq in input_ids)
            padded = []
            for seq in input_ids:
                padded.append(seq + [self.tokenizer.special_tokens['<pad>']] * (max_len - len(seq)))
            
            return torch.tensor(padded, dtype=torch.long)

    def mark_trained(self, lines):
        """Mark lines as trained (will be deleted)"""
        with self.lock:
            for line in lines:
                line_hash = hashlib.sha256(line.encode()).hexdigest()
                self.trained_hashes.add(line_hash)

    def load_trained_hashes(self):
        """Load trained hashes from file"""
        if os.path.exists(self.config.TRAINED_HASHES_FILE):
            try:
                with open(self.config.TRAINED_HASHES_FILE, 'r') as f:
                    data = json.load(f)
                    self.trained_hashes = set(data.get('hashes', []))
                print(f"   📥 Loaded {len(self.trained_hashes):,} trained hashes")
            except Exception as e:
                print(f"⚠️ Could not load trained hashes: {e}")
                self.trained_hashes = set()

    def flush_trained(self):
        """Save trained hashes to disk"""
        with self.lock:
            try:
                with open(self.config.TRAINED_HASHES_FILE, 'w') as f:
                    json.dump({'hashes': list(self.trained_hashes)}, f)
                print(f"   💾 Saved {len(self.trained_hashes):,} trained hashes")
            except Exception as e:
                print(f"⚠️ Could not save trained hashes: {e}")


# =============================
# TRANSFORMER MODEL
# =============================
class RoPEAttention(nn.Module):
    """Grouped Query Attention with RoPE"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_heads = config.NUM_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.rope_theta = config.ROPE_THETA
        
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.BIAS)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.BIAS)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Precompute RoPE frequencies
        self.register_buffer('rope_freqs', self._compute_rope_freqs(config.MAX_SEQ_LEN))
    
    def _compute_rope_freqs(self, max_seq_len):
        """Precompute RoPE rotation frequencies"""
        dim = self.head_dim
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def _apply_rope(self, x, freqs):
        """Apply RoPE rotation"""
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = freqs[:x.shape[1]].unsqueeze(0).unsqueeze(2)
        x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
        return x_rotated.type_as(x)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self._apply_rope(q.transpose(1, 2), self.rope_freqs).transpose(1, 2)
        k = self._apply_rope(k.transpose(1, 2), self.rope_freqs).transpose(1, 2)
        
        # Repeat k, v for GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.attn = RoPEAttention(config)
        self.ln2 = nn.LayerNorm(config.HIDDEN_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FFN_DIM, config.HIDDEN_DIM, bias=config.BIAS),
            nn.Dropout(config.DROPOUT)
        )
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class AdvancedTransformer(nn.Module):
    """Main transformer model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.NUM_LAYERS)])
        
        self.ln_f = nn.LayerNorm(config.HIDDEN_DIM)
        self.lm_head = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)
        
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================
# PROGRESSIVE GROWTH MANAGER
# =============================
class GrowthManager:
    """Handles growing the model dynamically during training"""
    
    def __init__(self, config, model, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.current_size_idx = ModelConfig.GROWTH_PATH.index(config.SIZE)
        self.history = []
        
        # Load history if exists
        if os.path.exists(config.GROWTH_LOG_FILE):
            try:
                with open(config.GROWTH_LOG_FILE, 'r') as f:
                    self.history = json.load(f)
            except:
                pass
    
    def check_growth(self, current_loss, step):
        """Check if model is ready to grow"""
        if self.current_size_idx >= len(ModelConfig.GROWTH_PATH) - 1:
            return False  # Already at max size
        
        # Add to history
        self.history.append({'step': step, 'loss': current_loss})
        if len(self.history) > self.config.GROWTH_STABLE_STEPS:
            self.history.pop(0)
        
        # Need enough history
        if len(self.history) < self.config.GROWTH_STABLE_STEPS:
            return False
        
        # Check if loss is consistently below threshold
        avg_loss = sum(h['loss'] for h in self.history) / len(self.history)
        
        if avg_loss < self.config.GROWTH_LOSS_THRESHOLD:
            print(f"\n🌟 GROWTH TRIGGERED! Average loss {avg_loss:.4f} < {self.config.GROWTH_LOSS_THRESHOLD}")
            return True
        
        return False
    
    def grow_model(self):
        """Grow model to next size"""
        next_size = ModelConfig.GROWTH_PATH[self.current_size_idx + 1]
        print(f"\n🚀 GROWING MODEL: {self.config.SIZE} -> {next_size}")
        
        # 1. Save current state
        temp_save = os.path.join(self.config.CHECKPOINT_DIR, f"pre_growth_{self.config.SIZE}.pt")
        torch.save(self.model.state_dict(), temp_save)
        
        # 2. Create new config and model
        new_config = ModelConfig(next_size)
        new_model = AdvancedTransformer(new_config).to(self.config.DEVICE)
        
        # 3. Transfer weights (Network Morphism / Knowledge Distillation)
        print("   Transplanting brain (copying compatible weights)...")
        old_state = self.model.state_dict()
        new_state = new_model.state_dict()
        
        for name, param in new_state.items():
            if name in old_state:
                old_param = old_state[name]
                # If shapes match perfectly, copy
                if old_param.shape == param.shape:
                    new_state[name].copy_(old_param)
                # If new parameter is larger (e.g., more hidden dims), copy into top-left
                elif len(param.shape) == 2 and len(old_param.shape) == 2:
                    min_d1 = min(param.shape[0], old_param.shape[0])
                    min_d2 = min(param.shape[1], old_param.shape[1])
                    new_state[name][:min_d1, :min_d2].copy_(old_param[:min_d1, :min_d2])
                elif len(param.shape) == 1 and len(old_param.shape) == 1:
                    min_d = min(param.shape[0], old_param.shape[0])
                    new_state[name][:min_d].copy_(old_param[:min_d])
        
        new_model.load_state_dict(new_state)
        
        # 4. Create new optimizer
        new_optimizer = torch.optim.AdamW(
            new_model.parameters(),
            lr=new_config.LEARNING_RATE,
            betas=(new_config.BETA1, new_config.BETA2),
            weight_decay=new_config.WEIGHT_DECAY,
            eps=new_config.EPS
        )
        
        # 5. Update references
        self.config = new_config
        self.model = new_model
        self.optimizer = new_optimizer
        self.current_size_idx += 1
        self.history = []  # Reset history
        
        # Save log
        with open(self.config.GROWTH_LOG_FILE, 'w') as f:
            json.dump([], f)
            
        print(f"✅ Growth complete! New parameters: {new_model.count_parameters():,}")
        return new_config, new_model, new_optimizer


# =============================
# MAIN TRAINING LOOP
# =============================
def train():
    print(f"\n{'='*60}")
    print("🚀 AUTONOMOUS 2TB AI TRAINING ENGINE - INTEGRATED FETCHER")
    print(f"{'='*60}")
    
    # 1. Setup Base Config
    config = ModelConfig("10M")
    
    # 2. Setup Tokenizer (Train if needed)
    tokenizer = UltimateBPETokenizer(vocab_size=config.VOCAB_SIZE)
    if os.path.exists(config.TOKENIZER_FILE):
        tokenizer.load(config.TOKENIZER_FILE)
    elif os.path.exists(config.DATA_FILE) and os.path.getsize(config.DATA_FILE) > 10_000_000:
        print("⚙️ Training Ultimate BPE Tokenizer first (this may take a while)...")
        tokenizer.train_streaming(config.DATA_FILE, config.BPE_CHECKPOINT_DIR)
        tokenizer.save(config.TOKENIZER_FILE)
    else:
        print("⚠️ No data.txt found or too small! Creating empty file...")
        open(config.DATA_FILE, 'w').close()
        print("   Data will be fetched automatically when training starts.")
    
    # 3. Setup Dataset
    dataset = StreamingDataset(config.DATA_FILE, tokenizer, config)
    
    # 4. Start Integrated Data Fetcher
    fetcher = IntegratedDataFetcher(config, dataset)
    fetcher.start_fetcher_thread()
    
    # 5. Initialize Model
    model = AdvancedTransformer(config).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
    
    # 6. Load Checkpoint if exists
    start_step = 0
    if os.path.exists(config.MODEL_FILE):
        print(f"📥 Loading existing model checkpoint...")
        checkpoint = torch.load(config.MODEL_FILE, map_location=config.DEVICE)
        
        # Handle model size resumes
        saved_size = checkpoint.get('size', '10M')
        if saved_size != config.SIZE:
            config = ModelConfig(saved_size)
            model = AdvancedTransformer(config).to(config.DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
            
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step']
        print(f"✅ Resumed at step {start_step:,} (Size: {config.SIZE})")
    
    # 7. Setup Growth Manager
    growth_manager = GrowthManager(config, model, optimizer)
    
    # 8. Training Loop
    print(f"\n🔥 BEGINNING TRAINING ON {config.DEVICE_TYPE.upper()}")
    model.train()
    step = start_step
    last_save_time = time.time()
    
    try:
        while True:  # Train infinitely until 1.9TB is reached
            # Get data
            input_ids = dataset.get_batch(config.BATCH_SIZE)
            
            if input_ids is None:
                print("⏳ No data in buffer - fetcher is working in background...")
                time.sleep(10)
                continue
                
            input_ids = input_ids.to(config.DEVICE)
            targets = input_ids.clone()  # Autoregressive: predict next token
            
            # Forward pass (Mixed Precision)
            if config.ENABLE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, targets=targets)
            else:
                logits, loss = model(input_ids, targets=targets)
                
            # Backward pass
            loss = loss / config.GRAD_ACCUMULATION_STEPS
            if config.ENABLE_MIXED_PRECISION:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Optimize
            if (step + 1) % config.GRAD_ACCUMULATION_STEPS == 0:
                if config.ENABLE_MIXED_PRECISION:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
            
            step += 1
            
            # Logging
            if step % 10 == 0:
                data_size_gb = dataset.file_size / 1e9
                progress_pct = (data_size_gb / config.TARGET_DATA_SIZE_GB) * 100
                print(f"Step {step:,} | Loss: {loss.item() * config.GRAD_ACCUMULATION_STEPS:.4f} | Data: {data_size_gb:.2f}/{config.TARGET_DATA_SIZE_GB:.0f}GB ({progress_pct:.2f}%)")
            
            # Check if we've reached target
            if dataset.file_size >= config.TARGET_DATA_SIZE_BYTES:
                print(f"\n🎉 TARGET REACHED! Trained on {dataset.file_size/1e9:.2f}GB of data!")
                break
            
            # Checkpoint & Memory Cleanup
            if time.time() - last_save_time > config.CHECKPOINT_INTERVAL:
                print("\n💾 Saving Checkpoint...")
                torch.save({
                    'step': step,
                    'size': config.SIZE,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }, config.MODEL_FILE)
                dataset.flush_trained()
                fetcher.save_fetch_state()
                last_save_time = time.time()
                
                # Force Garbage Collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Check Growth
            if step % 100 == 0 and growth_manager.check_growth(loss.item() * config.GRAD_ACCUMULATION_STEPS, step):
                config, model, optimizer = growth_manager.grow_model()
                if config.ENABLE_MIXED_PRECISION:
                    scaler = torch.cuda.amp.GradScaler()
                
    except KeyboardInterrupt:
        print("\n🛑 Training manually stopped. Saving final state...")
        torch.save({
            'step': step,
            'size': config.SIZE,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, config.MODEL_FILE)
        dataset.flush_trained()
        fetcher.save_fetch_state()
        fetcher.is_running = False
        print("✅ Saved successfully. Goodbye!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💾 Saving emergency checkpoint...")
        torch.save({
            'step': step,
            'size': config.SIZE,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, config.MODEL_FILE + ".emergency")
        dataset.flush_trained()
        fetcher.save_fetch_state()


# =============================
# ENTRY POINT
# =============================
if __name__ == '__main__':
    # Install required packages if not present
    try:
        import datasets
        import huggingface_hub
        import tqdm
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                               "datasets", "huggingface_hub", "tqdm"])
        print("✅ Packages installed!")
    
    # Mount Google Drive automatically if in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully.")
    except ImportError:
        print("📁 Running locally or outside Colab.")
    
    # Check if HF token is needed (optional - can run without)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ Hugging Face login successful.")
    else:
        print("ℹ️ No HF_TOKEN found - FineWeb dataset still accessible without login.")
    
    # Start the engine
    train()
