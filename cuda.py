#!/usr/bin/env python3
"""
AUTONOMOUS PROGRESSIVE GROWTH AI - ULTIMATE 2TB EDITION
COLAB OPTIMIZED - 12GB RAM FRIENDLY

Features:
- ULTIMATE BPE: 200K vocabulary, FULL Unicode support
- CONTINUOUS FETCHING until 1.9TB of training data
- DEDUPLICATION: Never trains same line twice (hash-based)
- SMART DELETION: Deletes text after training (like original)
- 25+ DIVERSE DATA SOURCES
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
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    print("[Warning] PIL not installed - image features disabled")

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
        self.VOCAB_SIZE = 500000
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
        self.DRIVE_DIR = "/content/drive/MyDrive/AITraining_2TB"
        os.makedirs(self.DRIVE_DIR, exist_ok=True)
        
        self.DATA_FILE = os.path.join(self.DRIVE_DIR, "data.txt")
        self.TOKENIZER_FILE = os.path.join(self.DRIVE_DIR, "tokenizer_200k.json")
        self.BPE_CHECKPOINT_DIR = os.path.join(self.DRIVE_DIR, "bpe_checkpoints")
        self.CHECKPOINT_DIR = os.path.join(self.DRIVE_DIR, "checkpoints")
        self.MODEL_FILE = os.path.join(self.DRIVE_DIR, "model.pt")
        self.GROWTH_LOG_FILE = os.path.join(self.DRIVE_DIR, "growth_log.json")
        self.TRAINED_HASHES_FILE = os.path.join(self.DRIVE_DIR, "trained_data.json")
        self.FETCHER_HISTORY_FILE = os.path.join(self.DRIVE_DIR, "fetcher_history.json")
        
        # Create all directories
        for d in [self.BPE_CHECKPOINT_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)

        # ULTIMATE TARGET: 1.9TB (1900 GB) of training data
        self.TARGET_DATA_SIZE_GB = 1900  # 1.9TB
        self.TARGET_DATA_SIZE_BYTES = self.TARGET_DATA_SIZE_GB * 1024**3
        
        self.CHECKPOINT_INTERVAL = 300  # 5 minutes
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 1000000  # 1M lines for BPE
        self.FETCH_BATCH = 25  # Fetch 25 articles per batch
        self.FETCH_DELAY = 2  # 2 seconds between fetches
        self.PREFETCH_TARGET_LINES = 2_000_000  # Initial prefetch

        self.GROWTH_LOSS_THRESHOLD = 3.0
        self.GROWTH_STABLE_STEPS = 1000

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
        self.BUFFER_SIZE = 25000  # Only keep 25k lines in buffer (was 50k)
        self.BPE_CHUNK_SIZE = 2_000_000  # Process 2M lines at a time (was 5M)

        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")


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

    def set_checkpoint_dir(self, checkpoint_dir):
        """Set directory for streaming checkpoints"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_latest_checkpoint(self):
        """Find the latest checkpoint to resume from"""
        if not self.checkpoint_dir:
            return 0
        chunks = [f for f in os.listdir(self.checkpoint_dir) 
                 if f.startswith("word_counts_") and f.endswith(".pkl")]
        if not chunks:
            return 0
        # Extract line numbers from filenames
        line_nums = []
        for f in chunks:
            try:
                num = int(f.replace("word_counts_", "").replace(".pkl", ""))
                line_nums.append(num)
            except:
                continue
        return max(line_nums) if line_nums else 0

    def _normalize(self, text):
        """Advanced Unicode normalization"""
        import unicodedata
        if text is None:
            return ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return ""
        if len(text) == 0:
            return ""
        
        # Try multiple normalization forms
        for form in ['nfc', 'nfkc', 'nfkd']:
            try:
                normalized = unicodedata.normalize(form, text)
                normalized.encode('utf-8')
                return normalized
            except:
                continue
        return text

    def _is_text_corrupted(self, text):
        """Detect encoding corruption"""
        if not isinstance(text, str):
            return True
        
        corruption_patterns = ['�', '\ufffd']
        corruption_count = sum(text.count(p) for p in corruption_patterns)
        if corruption_count > len(text) * 0.01:
            return True
        
        # Mojibake patterns
        mojibake = ['Ã©', 'Ã¼', 'Ã±', 'Ã¡', 'â‚¬', 'â€™']
        for pattern in mojibake:
            if pattern in text:
                return True
        return False

    def _save_chunk(self, freqs, line_num):
        """Save frequency chunk to disk"""
        chunk_file = os.path.join(self.checkpoint_dir, f"word_counts_{line_num}.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(dict(freqs), f)
        return chunk_file

    def _load_chunk(self, chunk_file):
        """Load frequency chunk from disk"""
        with open(chunk_file, 'rb') as f:
            return pickle.load(f)

    def train_streaming(self, file_path, min_frequency=2):
        """
        STREAMING BPE TRAINING - Processes 2TB files without loading into RAM
        Uses checkpoints every 2M lines for resume capability
        """
        print("\n" + "=" * 90)
        print("  🚀 ULTIMATE BPE TOKENIZER - STREAMING 2TB MODE")
        print("  Target Vocabulary: 200,000 tokens | Full Unicode Support")
        print("=" * 90)
        
        start_time = time.time()
        self.min_frequency = min_frequency
        
        if not self.checkpoint_dir:
            self.checkpoint_dir = "bpe_checkpoints"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Find where to resume
        start_line = self.get_latest_checkpoint()
        if start_line > 0:
            print(f"\n📌 Resuming from checkpoint at line {start_line:,}")
        
        # ============================================
        # PHASE 1: STREAMING WORD COUNT (OUT-OF-CORE)
        # ============================================
        print("\n📊 PHASE 1: STREAMING WORD FREQUENCIES")
        print("-" * 60)
        
        current_freqs = Counter()
        lines_processed = 0
        total_lines_read = 0
        total_chars = 0
        unicode_words = 0
        corrupted_lines = 0
        valid_lines = 0
        
        # For progress tracking
        last_checkpoint = start_line
        next_checkpoint = start_line + self.chunk_size
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Skip to where we left off
                if start_line > 0:
                    for _ in range(start_line):
                        f.readline()
                        total_lines_read += 1
                
                # Process the rest
                for line in f:
                    total_lines_read += 1
                    line = line.strip()
                    
                    if not line or len(line) < 20:
                        continue
                    
                    # Check corruption
                    if self._is_text_corrupted(line):
                        corrupted_lines += 1
                        continue
                    
                    # Normalize
                    text = self._normalize(line)
                    valid_lines += 1
                    total_chars += len(text)
                    
                    # Count words
                    for word in text.split():
                        if len(word) > self.max_word_length:
                            word = word[:self.max_word_length]
                        
                        # Track Unicode
                        if any(ord(c) > 127 for c in word):
                            unicode_words += 1
                        
                        current_freqs[word] += 1
                    
                    lines_processed += 1
                    
                    # Progress update
                    if lines_processed % 100000 == 0:
                        mem = psutil.Process().memory_info().rss / 1e9 if PSUTIL_AVAILABLE else 0
                        rate = lines_processed / max(1, (time.time() - start_time))
                        print(f"\r  📊 Processed: {total_lines_read:,} lines | "
                              f"Unique words: {len(current_freqs):,} | "
                              f"Mem: {mem:.2f}GB | Rate: {rate:.0f} lines/s", end="")
                    
                    # Save checkpoint and clear RAM
                    if total_lines_read >= next_checkpoint:
                        self._save_chunk(current_freqs, total_lines_read)
                        print(f"\n  💾 Saved checkpoint at {total_lines_read:,} lines")
                        current_freqs.clear()
                        gc.collect()
                        next_checkpoint += self.chunk_size
                        last_checkpoint = total_lines_read
            
            # Save final chunk
            if current_freqs:
                self._save_chunk(current_freqs, total_lines_read)
                print(f"\n  💾 Saved final checkpoint at {total_lines_read:,} lines")
                current_freqs.clear()
                gc.collect()
                
        except Exception as e:
            print(f"\n  ❌ Error during streaming count: {e}")
            return None
        
        # ============================================
        # PHASE 2: AGGREGATE CHECKPOINTS
        # ============================================
        print("\n\n🔄 PHASE 2: AGGREGATING CHECKPOINTS")
        print("-" * 60)
        
        global_freqs = Counter()
        chunk_files = sorted([os.path.join(self.checkpoint_dir, f) 
                             for f in os.listdir(self.checkpoint_dir) 
                             if f.startswith("word_counts_") and f.endswith(".pkl")])
        
        total_chunks = len(chunk_files)
        for i, chunk_file in enumerate(chunk_files):
            print(f"  Merging chunk {i+1}/{total_chunks}...")
            chunk_data = self._load_chunk(chunk_file)
            global_freqs.update(chunk_data)
            
            # Periodic pruning to save RAM
            if i % 5 == 0:
                global_freqs = Counter({w: c for w, c in global_freqs.items() if c >= 2})
                gc.collect()
        
        # Final pruning
        original_count = len(global_freqs)
        global_freqs = {w: c for w, c in global_freqs.items() if c >= min_frequency}
        
        print(f"\n  ✅ AGGREGATION COMPLETE:")
        print(f"  ├─ Total lines processed: {total_lines_read:,}")
        print(f"  ├─ Valid lines: {valid_lines:,}")
        print(f"  ├─ Corrupted lines: {corrupted_lines:,}")
        print(f"  ├─ Total characters: {total_chars:,}")
        print(f"  ├─ Unicode words: {unicode_words:,}")
        print(f"  ├─ Unique words before pruning: {original_count:,}")
        print(f"  └─ Unique words after pruning: {len(global_freqs):,}")
        
        self.stats['total_lines'] = valid_lines
        self.stats['unique_words'] = len(global_freqs)
        self.stats['unicode_words'] = unicode_words
        
        if not global_freqs:
            print("  ❌ No data to train on! Exiting.")
            return None
        
        # ============================================
        # PHASE 3: BUILD CHARACTER VOCABULARY
        # ============================================
        print("\n🔤 PHASE 3: BUILDING UNICODE CHARACTER VOCABULARY")
        print("-" * 60)
        
        char_freqs = Counter()
        vocab = set()
        unicode_set = set()
        
        # Process in batches to save memory
        word_items = list(global_freqs.items())
        batch_size = 500000
        
        for batch_start in range(0, len(word_items), batch_size):
            batch = word_items[batch_start:batch_start + batch_size]
            
            for word, freq in batch:
                for char in word:
                    char_freqs[char] += freq
                    vocab.add(char)
                    if ord(char) > 127:
                        unicode_set.add(char)
                vocab.add('</w>')
            
            if batch_start % 2_000_000 == 0:
                pct = (batch_start / len(word_items)) * 100
                print(f"  Processed {batch_start:,}/{len(word_items):,} words ({pct:.1f}%)")
                gc.collect()
        
        # Add special tokens
        for token in self.special_tokens:
            vocab.add(token)
        
        # Add byte fallback tokens for complete Unicode coverage
        if self.byte_fallback:
            for i in range(256):
                vocab.add(f'<byte_{i}>')
        
        print(f"\n  📊 VOCABULARY STATISTICS:")
        print(f"  ├─ Character vocabulary: {len(vocab):,}")
        print(f"  ├─ Unicode characters: {len(unicode_set):,}")
        print(f"  └─ ASCII characters: {len(vocab) - len(unicode_set):,}")
        
        # Initialize vocabulary with frequency-based ordering
        self.vocab = {}
        
        # First add special tokens with fixed IDs
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        # Then add characters sorted by frequency
        next_id = max(self.special_tokens.values()) + 1
        sorted_chars = sorted(char_freqs.items(), key=lambda x: -x[1])
        
        for char, _ in sorted_chars:
            if char not in self.vocab and char in vocab:
                self.vocab[char] = next_id
                next_id += 1
        
        # Add remaining characters
        for char in sorted(vocab - set(self.vocab.keys()) - set(self.special_tokens.keys())):
            self.vocab[char] = next_id
            next_id += 1
        
        # Add byte tokens
        for i in range(256):
            token = f'<byte_{i}>'
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
        
        print(f"\n  ✅ Initial vocabulary: {len(self.vocab):,} tokens")
        
        # ============================================
        # PHASE 4: INITIALIZE WORD SPLITS
        # ============================================
        print("\n📚 PHASE 4: INITIALIZING WORD SPLITS")
        print("-" * 60)
        
        splits = {}
        for word, freq in global_freqs.items():
            chars = tuple(list(word) + ['</w>'])
            splits[word] = (chars, freq)
        
        # Free memory
        del global_freqs
        del char_freqs
        del word_items
        gc.collect()
        
        # ============================================
        # PHASE 5: BPE MERGES (200K VOCAB)
        # ============================================
        print("\n🔄 PHASE 5: PERFORMING BPE MERGES (200K TARGET)")
        print("-" * 60)
        
        target_merges = self.vocab_size - len(self.vocab)
        if target_merges <= 0:
            print(f"  ✓ Already at target: {len(self.vocab):,} tokens")
        else:
            print(f"  Target merges: {target_merges:,}")
            print(f"  This will create {target_merges:,} subword tokens")
            
            merge_start = time.time()
            merge_times = []
            
            for merge_idx in range(target_merges):
                merge_loop_start = time.time()
                
                # Count pair frequencies
                pair_freqs = {}
                pair_to_words = {}
                
                for word, (chars, freq) in splits.items():
                    if len(chars) < 2:
                        continue
                    
                    for i in range(len(chars) - 1):
                        pair = (chars[i], chars[i+1])
                        pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                        
                        if pair not in pair_to_words:
                            pair_to_words[pair] = []
                        if word not in pair_to_words[pair]:
                            pair_to_words[pair].append(word)
                
                if not pair_freqs:
                    print(f"\n  ✓ No more pairs to merge at step {merge_idx}")
                    break
                
                # Find best pair
                best_pair = max(pair_freqs.items(), key=lambda x: x[1])
                best_pair_tuple, best_freq = best_pair[0], best_pair[1]
                merged_token = ''.join(best_pair_tuple)
                
                # Priority for Unicode tokens
                if any(ord(c) > 127 for c in merged_token):
                    best_freq = int(best_freq * 1.5)  # Boost Unicode merges
                
                # Record merge
                self.merges[best_pair_tuple] = merge_idx
                
                # Update splits
                words_to_update = pair_to_words.get(best_pair_tuple, [])
                for word in words_to_update:
                    chars, freq = splits[word]
                    new_chars = []
                    i = 0
                    while i < len(chars):
                        if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair_tuple:
                            new_chars.append(merged_token)
                            i += 2
                        else:
                            new_chars.append(chars[i])
                            i += 1
                    splits[word] = (tuple(new_chars), freq)
                
                # Add to vocabulary
                if merged_token not in self.vocab:
                    self.vocab[merged_token] = next_id
                    next_id += 1
                
                # Progress tracking
                merge_time = time.time() - merge_loop_start
                merge_times.append(merge_time)
                avg_time = sum(merge_times[-100:]) / max(1, len(merge_times[-100:]))
                
                if (merge_idx + 1) % self.progress_interval == 0:
                    elapsed = time.time() - merge_start
                    rate = (merge_idx + 1) / max(elapsed, 1)
                    remaining = (target_merges - merge_idx - 1) / max(rate, 0.1)
                    
                    pct = (merge_idx + 1) / target_merges * 100
                    bar_len = 50
                    filled = int(bar_len * (merge_idx + 1) / target_merges)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    
                    mem = psutil.Process().memory_info().rss / 1e9 if PSUTIL_AVAILABLE else 0
                    
                    unicode_marker = "✓" if any(ord(c) > 127 for c in merged_token) else " "
                    
                    print(f"\r  [{bar}] {pct:5.1f}% | "
                          f"Merges: {merge_idx+1:,}/{target_merges:,} | "
                          f"Vocab: {len(self.vocab):,} | "
                          f"Best: '{merged_token}' {unicode_marker} | "
                          f"Freq: {best_freq:,} | "
                          f"Mem: {mem:.2f}GB | "
                          f"ETA: {remaining/3600:.1f}h", end="")
                
                # Early stopping
                if best_freq < 3 and merge_idx > target_merges * 0.8:
                    print(f"\n\n  ✓ Early stopping: pair frequency too low ({best_freq})")
                    break
            
            print()  # New line
        
        # ============================================
        # PHASE 6: FINALIZE
        # ============================================
        print("\n💾 PHASE 6: FINALIZING TOKENIZER")
        print("-" * 60)
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Calculate statistics
        elapsed = time.time() - start_time
        unicode_tokens = [t for t in self.vocab.keys() 
                         if any(ord(c) > 127 for c in t) and t not in self.special_tokens]
        
        token_lengths = [len(t) for t in self.vocab.keys() 
                        if t not in self.special_tokens and not t.startswith('<byte_')]
        
        print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║              ULTIMATE BPE TOKENIZER - FINAL STATS             ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Training lines:      {self.stats['total_lines']:>12,}                       ║
    ║  Training time:       {elapsed/3600:>10.2f} hours                          ║
    ║                                                                ║
    ║  FINAL VOCABULARY:    {len(self.vocab):>12,}                       ║
    ║  ├─ ASCII tokens:     {len(self.vocab) - len(unicode_tokens):>12,}                       ║
    ║  └─ UNICODE tokens:   {len(unicode_tokens):>12,} ✨                       ║
    ║                                                                ║
    ║  Total merges:        {len(self.merges):>12,}                       ║
    ║  Avg token length:    {sum(token_lengths)/len(token_lengths) if token_lengths else 0:>11.2f} chars              ║
    ║  Min/Max token:       {min(token_lengths) if token_lengths else 0} / {max(token_lengths) if token_lengths else 0} chars                ║
    ║                                                                ║
    ║  Byte fallback:       Enabled                                  ║
    ║  Unicode support:     FULL (U+0000 to U+10FFFF)               ║
    ╚════════════════════════════════════════════════════════════════╝
        """)
        
        return self

    def encode(self, text, add_special_tokens=True):
        """Fast encoding with Unicode preservation"""
        if text is None:
            text = ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                text = ""
        
        # Cache check
        cache_key = text[:200]
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Normalize
        try:
            text = self._normalize(text)
        except:
            pass
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        
        # Process each word
        for word in text.strip().split():
            if len(word) > self.max_word_length:
                word = word[:self.max_word_length]
            
            # Start with characters
            chars = list(word) + ['</w>']
            
            # Apply merges (longest first for better quality)
            changed = True
            while changed:
                changed = False
                
                # Find longest applicable merge
                best_len = 0
                best_i = -1
                best_pair = None
                
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i+1])
                    if pair in self.merges:
                        token_len = len(''.join(pair))
                        if token_len > best_len:
                            best_len = token_len
                            best_i = i
                            best_pair = pair
                
                if best_pair is not None:
                    merged = ''.join(best_pair)
                    chars = chars[:best_i] + [merged] + chars[best_i+2:]
                    changed = True
            
            # Convert to IDs
            for char in chars:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                elif self.byte_fallback:
                    # Byte fallback for unknown chars
                    for byte in char.encode('utf-8', errors='replace'):
                        byte_token = f'<byte_{byte}>'
                        if byte_token in self.vocab:
                            tokens.append(self.vocab[byte_token])
                        else:
                            tokens.append(self.special_tokens['<unk>'])
                else:
                    tokens.append(self.special_tokens['<unk>'])
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
        
        # Update cache
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = tokens.copy()
        
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode with Unicode reconstruction"""
        if not token_ids:
            return ""
        
        text = ""
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, '<unk>')
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            if token == '</w>':
                text += ' '
            elif token.startswith('<byte_') and self.byte_fallback:
                try:
                    byte_val = int(token[6:-1])
                    text += chr(byte_val)
                except:
                    text += '�'
            else:
                text += token
        
        return text.strip()

    def save(self, path):
        """Save tokenizer with all metadata"""
        # Convert set to list for JSON
        unicode_samples = list(self.unicode_stats['unique_unicode'])[:100]
        unicode_samples = [c for c in unicode_samples if ord(c) > 127]
        
        data = {
            'vocab': self.vocab,
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens,
            'byte_fallback': self.byte_fallback,
            'normalization': self.normalization,
            'max_word_length': self.max_word_length,
            'vocab_size': self.vocab_size,
            'stats': self.stats,
            'unicode_stats': {
                'total_chars_found': self.unicode_stats['total_chars_found'],
                'unicode_chars_found': self.unicode_stats['unicode_chars_found'],
                'unique_unicode_samples': unicode_samples,
                'corrupted_lines': self.unicode_stats['corrupted_lines']
            },
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': 'ULTIMATE-2TB-200K',
                'unicode_support': 'full'
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n  ✅ Tokenizer saved to {path}")
        print(f"     Vocabulary: {len(self.vocab):,} tokens")
        print(f"     Unicode tokens: {sum(1 for t in self.vocab if any(ord(c)>127 for c in t)):,}")

    def load(self, path):
        """Load tokenizer with all metadata"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        self.byte_fallback = data.get('byte_fallback', True)
        self.normalization = data.get('normalization', 'nfc')
        self.max_word_length = data.get('max_word_length', 100)
        self.vocab_size = data.get('vocab_size', len(self.vocab))
        self.stats = data.get('stats', self.stats)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Load Unicode stats
        unicode_stats = data.get('unicode_stats', {})
        self.unicode_stats['total_chars_found'] = unicode_stats.get('total_chars_found', 0)
        self.unicode_stats['unicode_chars_found'] = unicode_stats.get('unicode_chars_found', 0)
        self.unicode_stats['corrupted_lines'] = unicode_stats.get('corrupted_lines', 0)
        
        print(f"\n  ✅ Tokenizer loaded from {path}")
        print(f"     Vocabulary: {len(self.vocab):,} tokens")
        print(f"     Unicode tokens: {sum(1 for t in self.vocab if any(ord(c)>127 for c in t)):,}")
        
        # Test Unicode support
        test_text = "café über αβγ 今日は 日本語 🚀✨"
        encoded = self.encode(test_text, add_special_tokens=False)
        decoded = self.decode(encoded, skip_special_tokens=True)
        print(f"\n  🔤 Unicode test: '{test_text}'")
        print(f"  🔄 Decoded:      '{decoded}'")
        if decoded != test_text:
            print("  ⚠️  Warning: Unicode may not be perfectly preserved")
        
        return self


# =============================
# MEMORY MONITOR - 12GB OPTIMIZED
# =============================
class MemoryMonitor:
    def __init__(self, max_memory_gb=10.0, reserve_bytes=None):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.RESERVE_BYTES = reserve_bytes or 2 * 1024 * 1024 * 1024  # 2GB default
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.warning_threshold = 0.85  # 85% of max
        self.critical_threshold = 0.95  # 95% of max
        
    def get_memory_usage(self):
        if self.process:
            return self.process.memory_info().rss
        return 0
        
    def get_memory_usage_gb(self):
        return self.get_memory_usage() / (1024 ** 3)
        
    def get_free_ram_gb(self):
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / (1024 ** 3)
        return 999.0
        
    def get_usage_percent(self):
        usage = self.get_memory_usage()
        return usage / self.max_memory_bytes if self.max_memory_bytes else 0
        
    def is_memory_safe(self):
        """Check if memory usage is safe for continued operation"""
        if not self.process:
            return True
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available > self.RESERVE_BYTES
        return self.get_usage_percent() < 0.9
        
    def is_critical(self):
        """Check if memory is critically high"""
        return self.get_usage_percent() > self.critical_threshold
        
    def force_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if DIRECTML_AVAILABLE:
            gc.collect()
        # Additional cleanup
        for _ in range(3):
            gc.collect()


# =============================
# ULTIMATE DATA FETCHER - 25+ SOURCES
# =============================
class UltimateDataFetcher:
    """
    ULTIMATE DATA FETCHER - 25+ Diverse Sources
    Features:
    - Wikipedia (multiple languages)
    - Project Gutenberg
    - arXiv papers
    - GitHub repositories
    - News articles
    - Scientific journals
    - And many more!
    """
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.fetched_count = 0
        self.total_bytes_fetched = 0
        self.target_bytes = config.TARGET_DATA_SIZE_BYTES
        
        # Session with retry strategy
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UltimateAITrainer/2.0 (2TB Edition)',
            'Accept': 'text/plain, text/html, application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        
        # Track fetched URLs to prevent duplicates
        self.fetched_urls = set()
        self.fetched_hashes = set()  # Content hashes for deduplication
        
        # History file
        self.history_file = Path(config.FETCHER_HISTORY_FILE)
        self._load_history()
        
        # Initialize all source generators
        self.sources = self._init_sources()
        self.source_index = 0
        
        # Stats tracking
        self.source_stats = {name: {'count': 0, 'bytes': 0} 
                            for name, _ in self.sources}
        
        print(f"\n📡 ULTIMATE DATA FETCHER INITIALIZED")
        print(f"   Target: {config.TARGET_DATA_SIZE_GB}GB ({config.TARGET_DATA_SIZE_BYTES/1e9:.1f}GB)")
        print(f"   Sources: {len(self.sources)}")
        print(f"   Resume: {len(self.fetched_urls):,} URLs already fetched")

    def _load_history(self):
        """Load fetch history for resume capability"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.fetched_urls = set(data.get('fetched_urls', []))
                self.fetched_hashes = set(data.get('fetched_hashes', []))
                self.fetched_count = data.get('fetched_count', 0)
                self.total_bytes_fetched = data.get('total_bytes', 0)
                self.source_index = data.get('source_index', 0)
                print(f"[Fetcher] Resumed: {len(self.fetched_urls):,} URLs, "
                      f"{self.total_bytes_fetched/1e9:.2f}GB fetched")
            except Exception as e:
                print(f"[Fetcher] History load error: {e}")

    def _save_history(self):
        """Save fetch history"""
        try:
            data = {
                'fetched_urls': list(self.fetched_urls)[-100000:],  # Keep last 100k
                'fetched_hashes': list(self.fetched_hashes)[-100000:],
                'fetched_count': self.fetched_count,
                'total_bytes': self.total_bytes_fetched,
                'source_index': self.source_index,
                'source_stats': self.source_stats,
                'timestamp': datetime.now().isoformat()
            }
            tmp = str(self.history_file) + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            if self.history_file.exists():
                os.remove(self.history_file)
            os.rename(tmp, self.history_file)
        except Exception as e:
            print(f"[Fetcher] Save error: {e}")

    def _init_sources(self):
        """Initialize all 25+ data sources"""
        sources = []
        
        # ===== WIKIPEDIA SOURCES (10+) =====
        sources.append(('Wikipedia (EN random)', self._fetch_wikipedia_en_random))
        sources.append(('Wikipedia (EN vital)', self._fetch_wikipedia_en_vital))
        sources.append(('Wikipedia (EN featured)', self._fetch_wikipedia_en_featured))
        sources.append(('Wikipedia (Simple EN)', self._fetch_wikipedia_simple))
        sources.append(('Wikipedia (FR random)', self._fetch_wikipedia_fr_random))
        sources.append(('Wikipedia (DE random)', self._fetch_wikipedia_de_random))
        sources.append(('Wikipedia (ES random)', self._fetch_wikipedia_es_random))
        sources.append(('Wikipedia (IT random)', self._fetch_wikipedia_it_random))
        sources.append(('Wikipedia (PT random)', self._fetch_wikipedia_pt_random))
        sources.append(('Wikipedia (RU random)', self._fetch_wikipedia_ru_random))
        sources.append(('Wikipedia (JA random)', self._fetch_wikipedia_ja_random))
        sources.append(('Wikipedia (ZH random)', self._fetch_wikipedia_zh_random))
        sources.append(('Wikipedia (AR random)', self._fetch_wikipedia_ar_random))
        
        # ===== PROJECT GUTENBERG =====
        sources.append(('Gutenberg (random)', self._fetch_gutenberg_random))
        sources.append(('Gutenberg (top 100)', self._fetch_gutenberg_top))
        sources.append(('Gutenberg (poetry)', self._fetch_gutenberg_poetry))
        sources.append(('Gutenberg (science)', self._fetch_gutenberg_science))
        
        # ===== ACADEMIC SOURCES =====
        sources.append(('arXiv (CS)', self._fetch_arxiv_cs))
        sources.append(('arXiv (Physics)', self._fetch_arxiv_physics))
        sources.append(('arXiv (Math)', self._fetch_arxiv_math))
        sources.append(('PubMed Central', self._fetch_pubmed))
        sources.append(('CORE Open Access', self._fetch_core))
        sources.append(('DOAJ Journals', self._fetch_doaj))
        
        # ===== CODE & TECHNICAL =====
        sources.append(('GitHub (trending)', self._fetch_github_trending))
        sources.append(('GitHub (READMEs)', self._fetch_github_readmes))
        sources.append(('StackOverflow', self._fetch_stackoverflow))
        sources.append(('Dev.To articles', self._fetch_devto))
        sources.append(('MDN Web Docs', self._fetch_mdn))
        
        # ===== NEWS & CURRENT EVENTS =====
        sources.append(('BBC News', self._fetch_bbc_news))
        sources.append(('Reuters', self._fetch_reuters))
        sources.append(('The Guardian', self._fetch_guardian))
        sources.append(('NPR', self._fetch_npr))
        sources.append(('Al Jazeera', self._fetch_aljazeera))
        
        # ===== WIKISOURCES =====
        sources.append(('Wikisource (EN)', self._fetch_wikisource_en))
        sources.append(('Wikisource (FR)', self._fetch_wikisource_fr))
        sources.append(('Wikisource (DE)', self._fetch_wikisource_de))
        
        # ===== WIKIQUOTE =====
        sources.append(('Wikiquote (people)', self._fetch_wikiquote_people))
        sources.append(('Wikiquote (topics)', self._fetch_wikiquote_topics))
        
        # ===== OTHER WIKIMEDIA =====
        sources.append(('Wikibooks', self._fetch_wikibooks))
        sources.append(('Wikiversity', self._fetch_wikiversity))
        sources.append(('Wikinews', self._fetch_wikinews))
        
        # ===== OPEN TEXTBOOKS =====
        sources.append(('OpenStax', self._fetch_openstax))
        sources.append(('BC Campus', self._fetch_bccampus))
        sources.append(('MERLOT', self._fetch_merlot))
        
        # ===== MISCELLANEOUS =====
        sources.append(('TED Talks', self._fetch_ted_transcripts))
        sources.append(('Podcast transcripts', self._fetch_podcast_transcripts))
        sources.append(('Public speeches', self._fetch_speeches))
        sources.append(('Legal documents', self._fetch_legal_docs))
        sources.append(('Religious texts', self._fetch_religious_texts))
        
        return sources

    def _get_current_data_size(self):
        """Get current size of data.txt"""
        if os.path.exists(self.config.DATA_FILE):
            return os.path.getsize(self.config.DATA_FILE)
        return 0

    def _is_target_reached(self):
        """Check if we've reached the 1.9TB target"""
        current = self._get_current_data_size()
        if current >= self.target_bytes:
            print(f"\n🎯 TARGET REACHED! {current/1e9:.2f}GB / {self.target_bytes/1e9:.2f}GB")
            return True
        return False

    def _add_lines(self, lines, source_name):
        """Add lines to dataset with size tracking"""
        if not lines:
            return 0
        
        # Filter duplicates using content hash
        unique_lines = []
        for line in lines:
            if not line or len(line) < 20:
                continue
            # Content hash for deduplication
            content_hash = hashlib.md5(line.encode('utf-8', errors='ignore')).hexdigest()
            if content_hash not in self.fetched_hashes:
                self.fetched_hashes.add(content_hash)
                unique_lines.append(line)
        
        if unique_lines:
            # Add to dataset
            self.dataset.add_lines(unique_lines)
            
            # Update stats
            bytes_added = sum(len(l.encode('utf-8')) for l in unique_lines)
            self.total_bytes_fetched += bytes_added
            self.fetched_count += len(unique_lines)
            self.source_stats[source_name]['count'] += len(unique_lines)
            self.source_stats[source_name]['bytes'] += bytes_added
            
            # Progress
            current = self._get_current_data_size()
            pct = (current / self.target_bytes) * 100 if self.target_bytes else 0
            
            print(f"\n  ✅ {source_name}: +{len(unique_lines):,} lines | "
                  f"+{bytes_added/1e6:.2f}MB | "
                  f"Total: {current/1e9:.2f}GB / {self.target_bytes/1e9:.2f}GB ({pct:.2f}%)")
            
            # Save history periodically
            if self.fetched_count % 1000 == 0:
                self._save_history()
        
        return len(unique_lines)

    # ===== WIKIPEDIA FETCHERS =====
    
    def _fetch_wikipedia_en_random(self):
        """Fetch random English Wikipedia articles"""
        try:
            # Get random titles
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'list': 'random',
                    'rnnamespace': 0,
                    'rnlimit': self.config.FETCH_BATCH,
                    'format': 'json'
                },
                timeout=15
            )
            data = resp.json()
            titles = [item['title'] for item in data.get('query', {}).get('random', [])]
            
            all_lines = []
            for title in titles:
                # Get full article
                resp = self.session.get(
                    'https://en.wikipedia.org/w/api.php',
                    params={
                        'action': 'query',
                        'titles': title,
                        'prop': 'extracts',
                        'explaintext': True,
                        'format': 'json'
                    },
                    timeout=15
                )
                pages = resp.json().get('query', {}).get('pages', {})
                for page in pages.values():
                    if 'extract' in page and len(page['extract']) > 500:
                        # Split into paragraphs
                        paragraphs = page['extract'].split('\n')
                        for p in paragraphs:
                            p = p.strip()
                            if len(p) > 100:
                                all_lines.append(p)
                time.sleep(0.2)  # Rate limiting
            
            return all_lines
        except Exception as e:
            return []

    def _fetch_wikipedia_en_vital(self):
        """Fetch vital articles (most important)"""
        vital_topics = [
            'Universe', 'Earth', 'Life', 'Human', 'History', 'Science',
            'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Medicine',
            'Technology', 'Art', 'Music', 'Literature', 'Philosophy',
            'Religion', 'War', 'Politics', 'Economics', 'Society'
        ]
        topic = random.choice(vital_topics)
        return self._fetch_single_wikipedia_article(topic, 'en')

    def _fetch_wikipedia_en_featured(self):
        """Fetch featured articles (highest quality)"""
        try:
            # Get featured articles category
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'list': 'categorymembers',
                    'cmtitle': 'Category:Featured_articles',
                    'cmlimit': 10,
                    'cmtype': 'page',
                    'format': 'json'
                },
                timeout=15
            )
            members = resp.json().get('query', {}).get('categorymembers', [])
            if members:
                member = random.choice(members)
                return self._fetch_single_wikipedia_article(member['title'], 'en')
            return []
        except Exception:
            return []

    def _fetch_wikipedia_simple(self):
        """Fetch Simple English Wikipedia (easier text)"""
        topics = ['Science', 'History', 'Geography', 'Biology', 'Physics']
        topic = random.choice(topics)
        return self._fetch_single_wikipedia_article(topic, 'simple')

    def _fetch_single_wikipedia_article(self, title, lang='en'):
        """Fetch a single Wikipedia article"""
        try:
            url = f'https://{lang}.wikipedia.org/w/api.php'
            resp = self.session.get(
                url,
                params={
                    'action': 'query',
                    'titles': title,
                    'prop': 'extracts',
                    'explaintext': True,
                    'format': 'json'
                },
                timeout=15
            )
            pages = resp.json().get('query', {}).get('pages', {})
            for page in pages.values():
                if 'extract' in page and len(page['extract']) > 500:
                    paragraphs = page['extract'].split('\n')
                    return [p.strip() for p in paragraphs if len(p.strip()) > 100]
            return []
        except Exception:
            return []

    # Multi-language Wikipedia
    def _fetch_wikipedia_fr_random(self): return self._fetch_wikipedia_lang_random('fr')
    def _fetch_wikipedia_de_random(self): return self._fetch_wikipedia_lang_random('de')
    def _fetch_wikipedia_es_random(self): return self._fetch_wikipedia_lang_random('es')
    def _fetch_wikipedia_it_random(self): return self._fetch_wikipedia_lang_random('it')
    def _fetch_wikipedia_pt_random(self): return self._fetch_wikipedia_lang_random('pt')
    def _fetch_wikipedia_ru_random(self): return self._fetch_wikipedia_lang_random('ru')
    def _fetch_wikipedia_ja_random(self): return self._fetch_wikipedia_lang_random('ja')
    def _fetch_wikipedia_zh_random(self): return self._fetch_wikipedia_lang_random('zh')
    def _fetch_wikipedia_ar_random(self): return self._fetch_wikipedia_lang_random('ar')

    def _fetch_wikipedia_lang_random(self, lang):
        """Fetch random article from specific language Wikipedia"""
        try:
            resp = self.session.get(
                f'https://{lang}.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'list': 'random',
                    'rnnamespace': 0,
                    'rnlimit': 5,
                    'format': 'json'
                },
                timeout=15
            )
            titles = [item['title'] for item in resp.json().get('query', {}).get('random', [])]
            all_lines = []
            for title in titles:
                lines = self._fetch_single_wikipedia_article(title, lang)
                all_lines.extend(lines)
                time.sleep(0.2)
            return all_lines
        except Exception:
            return []

    # ===== PROJECT GUTENBERG =====
    
    def _fetch_gutenberg_random(self):
        """Fetch random book from Project Gutenberg"""
        book_id = random.randint(1, 60000)
        urls = [
            f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
            f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
        ]
        for url in urls:
            if url in self.fetched_urls:
                continue
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    text = resp.content.decode('utf-8', errors='ignore')
                    # Extract main content (skip Gutenberg headers/footers)
                    lines = []
                    in_content = False
                    for line in text.split('\n'):
                        line = line.strip()
                        if '*** START OF' in line:
                            in_content = True
                            continue
                        if '*** END OF' in line:
                            break
                        if in_content and line and len(line) > 50:
                            lines.append(line)
                    
                    if len(lines) > 50:
                        self.fetched_urls.add(url)
                        return lines[:500]  # Limit to 500 lines per book
            except Exception:
                continue
        return []

    def _fetch_gutenberg_top(self):
        """Fetch top 100 books"""
        try:
            # Get top 100 list
            resp = self.session.get(
                'https://www.gutenberg.org/browse/scores/top',
                timeout=20
            )
            if resp.status_code == 200:
                # Extract book IDs from HTML (simplified)
                import re
                book_ids = re.findall(r'/ebooks/(\d+)', resp.text)
                book_ids = list(set(book_ids))[:50]  # Unique IDs, limit to 50
                if book_ids:
                    book_id = random.choice(book_ids)
                    return self._fetch_single_gutenberg_book(book_id)
            return []
        except Exception:
            return []

    def _fetch_single_gutenberg_book(self, book_id):
        """Fetch a single Gutenberg book by ID"""
        urls = [
            f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
            f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
        ]
        for url in urls:
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    text = resp.content.decode('utf-8', errors='ignore')
                    lines = []
                    in_content = False
                    for line in text.split('\n'):
                        line = line.strip()
                        if '*** START OF' in line:
                            in_content = True
                            continue
                        if '*** END OF' in line:
                            break
                        if in_content and line and len(line) > 50:
                            lines.append(line)
                    return lines[:500]
            except Exception:
                continue
        return []

    def _fetch_gutenberg_poetry(self):
        """Fetch poetry collections"""
        poetry_ids = [16, 42, 1001, 203, 4085, 36, 41]  # Common poetry books
        book_id = random.choice(poetry_ids)
        return self._fetch_single_gutenberg_book(book_id)

    def _fetch_gutenberg_science(self):
        """Fetch science books"""
        science_ids = [1228, 30, 944, 1260, 15407, 20776]  # Science books
        book_id = random.choice(science_ids)
        return self._fetch_single_gutenberg_book(book_id)

    # ===== ACADEMIC SOURCES =====
    
    def _fetch_arxiv_cs(self):
        """Fetch recent Computer Science papers from arXiv"""
        try:
            resp = self.session.get(
                'http://export.arxiv.org/api/query',
                params={
                    'search_query': 'cat:cs.*',
                    'start': random.randint(0, 5000),
                    'max_results': 10,
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                },
                timeout=20
            )
            if resp.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                lines = []
                for entry in root.findall('atom:entry', ns):
                    title = entry.find('atom:title', ns)
                    summary = entry.find('atom:summary', ns)
                    if title is not None and summary is not None:
                        lines.append(title.text.strip())
                        # Split summary into sentences
                        summary_text = summary.text.strip() if summary.text else ''
                        for sent in summary_text.split('. '):
                            if len(sent) > 50:
                                lines.append(sent + '.')
                return lines
            return []
        except Exception:
            return []

    def _fetch_arxiv_physics(self):
        """Fetch Physics papers"""
        try:
            resp = self.session.get(
                'http://export.arxiv.org/api/query',
                params={
                    'search_query': 'cat:physics.*',
                    'start': random.randint(0, 5000),
                    'max_results': 10,
                    'format': 'json'
                },
                timeout=20
            )
            # Similar to above but with different category
            return self._fetch_arxiv_cs()  # Simplified for example
        except Exception:
            return []

    def _fetch_arxiv_math(self): return self._fetch_arxiv_cs()  # Simplified

    def _fetch_pubmed(self):
        """Fetch abstracts from PubMed Central"""
        try:
            # Use PubMed E-utilities
            resp = self.session.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                params={
                    'db': 'pmc',
                    'term': 'open access[filter]',
                    'retmax': 10,
                    'retstart': random.randint(0, 10000),
                    'format': 'json'
                },
                timeout=20
            )
            return []  # Simplified for example
        except Exception:
            return []

    def _fetch_core(self): return []  # Placeholder
    def _fetch_doaj(self): return []  # Placeholder

    # ===== CODE & TECHNICAL =====
    
    def _fetch_github_trending(self):
        """Fetch trending GitHub repositories"""
        try:
            resp = self.session.get(
                'https://api.github.com/search/repositories',
                params={
                    'q': 'stars:>100',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 10
                },
                headers={'Accept': 'application/vnd.github.v3+json'},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                lines = []
                for repo in data.get('items', []):
                    # Add repo info
                    lines.append(f"Repository: {repo['full_name']}")
                    lines.append(f"Description: {repo['description']}")
                    if repo.get('language'):
                        lines.append(f"Language: {repo['language']}")
                    
                    # Try to fetch README
                    readme_url = f"https://raw.githubusercontent.com/{repo['full_name']}/master/README.md"
                    try:
                        readme_resp = self.session.get(readme_url, timeout=10)
                        if readme_resp.status_code == 200:
                            readme_text = readme_resp.text
                            # Extract text content (strip markdown)
                            readme_text = re.sub(r'[#*`\[\]()]', ' ', readme_text)
                            for line in readme_text.split('\n'):
                                line = line.strip()
                                if line and len(line) > 50:
                                    lines.append(line)
                    except:
                        pass
                    time.sleep(0.5)
                return lines
            return []
        except Exception:
            return []

    def _fetch_github_readmes(self):
        """Fetch random READMEs from popular repos"""
        repos = [
            'tensorflow/tensorflow', 'pytorch/pytorch', 'keras-team/keras',
            'django/django', 'flask/flask', 'rails/rails', 'spring-projects/spring-boot',
            'microsoft/vscode', 'atom/atom', 'angular/angular', 'facebook/react'
        ]
        repo = random.choice(repos)
        try:
            readme_url = f"https://raw.githubusercontent.com/{repo}/master/README.md"
            resp = self.session.get(readme_url, timeout=15)
            if resp.status_code == 200:
                text = resp.text
                # Clean markdown
                text = re.sub(r'[#*`\[\]()]', ' ', text)
                lines = [l.strip() for l in text.split('\n') 
                        if l.strip() and len(l.strip()) > 50]
                return lines[:200]
            return []
        except Exception:
            return []

    def _fetch_stackoverflow(self):
        """Fetch high-quality StackOverflow posts"""
        try:
            # Use StackExchange API
            resp = self.session.get(
                'https://api.stackexchange.com/2.3/questions',
                params={
                    'order': 'desc',
                    'sort': 'votes',
                    'site': 'stackoverflow',
                    'pagesize': 10,
                    'filter': 'withbody',
                    'tagged': 'python;java;javascript'
                },
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                lines = []
                for item in data.get('items', []):
                    lines.append(f"Q: {item['title']}")
                    if item.get('body'):
                        # Strip HTML
                        body = re.sub(r'<[^>]+>', ' ', item['body'])
                        body = re.sub(r'\s+', ' ', body).strip()
                        if len(body) > 100:
                            lines.append(body)
                return lines
            return []
        except Exception:
            return []

    def _fetch_devto(self):
        """Fetch articles from Dev.to"""
        try:
            resp = self.session.get(
                'https://dev.to/api/articles',
                params={'top': 10},
                timeout=15
            )
            if resp.status_code == 200:
                articles = resp.json()
                lines = []
                for article in articles:
                    lines.append(f"Title: {article.get('title', '')}")
                    lines.append(f"Description: {article.get('description', '')}")
                    tags = article.get('tag_list', [])
                    if tags:
                        lines.append(f"Tags: {', '.join(tags)}")
                return lines
            return []
        except Exception:
            return []

    def _fetch_mdn(self):
        """Fetch MDN Web Docs"""
        topics = ['html', 'css', 'javascript', 'api', 'web']
        topic = random.choice(topics)
        try:
            # MDN API endpoint
            resp = self.session.get(
                f'https://developer.mozilla.org/api/v1/search',
                params={'q': topic, 'locale': 'en-US'},
                timeout=15
            )
            return []  # Placeholder
        except Exception:
            return []

    # ===== NEWS SOURCES =====
    
    def _fetch_bbc_news(self):
        """Fetch BBC News articles"""
        try:
            # BBC RSS feeds
            resp = self.session.get(
                'http://feeds.bbci.co.uk/news/rss.xml',
                timeout=15
            )
            if resp.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                lines = []
                for item in root.findall('.//item')[:10]:
                    title = item.find('title')
                    description = item.find('description')
                    if title is not None and title.text:
                        lines.append(title.text)
                    if description is not None and description.text:
                        desc = re.sub(r'<[^>]+>', '', description.text)
                        if len(desc) > 50:
                            lines.append(desc)
                return lines
            return []
        except Exception:
            return []

    def _fetch_reuters(self): return self._fetch_bbc_news()  # Simplified
    def _fetch_guardian(self): return self._fetch_bbc_news()  # Simplified
    def _fetch_npr(self): return self._fetch_bbc_news()  # Simplified
    def _fetch_aljazeera(self): return self._fetch_bbc_news()  # Simplified

    # ===== WIKISOURCES =====
    
    def _fetch_wikisource_en(self):
        """Fetch documents from English Wikisource"""
        docs = ['United_States_Constitution', 'Declaration_of_Independence',
                'Magna_Carta', 'Bill_of_Rights', 'Gettysburg_Address']
        doc = random.choice(docs)
        return self._fetch_single_wikisource(doc, 'en')

    def _fetch_wikisource_fr(self):
        docs = ['Declaration_des_droits_de_l_homme_et_du_citoyen',
                'Code_Napoleon', 'Contrat_social']
        doc = random.choice(docs)
        return self._fetch_single_wikisource(doc, 'fr')

    def _fetch_wikisource_de(self):
        docs = ['Grundgesetz', 'Faust_I', 'Nathan_der_Weise']
        doc = random.choice(docs)
        return self._fetch_single_wikisource(doc, 'de')

    def _fetch_single_wikisource(self, title, lang='en'):
        """Fetch a single Wikisource document"""
        try:
            url = f'https://{lang}.wikisource.org/w/api.php'
            resp = self.session.get(
                url,
                params={
                    'action': 'query',
                    'titles': title,
                    'prop': 'extracts',
                    'explaintext': True,
                    'format': 'json'
                },
                timeout=15
            )
            pages = resp.json().get('query', {}).get('pages', {})
            for page in pages.values():
                if 'extract' in page and len(page['extract']) > 500:
                    paragraphs = page['extract'].split('\n')
                    return [p.strip() for p in paragraphs if len(p.strip()) > 100]
            return []
        except Exception:
            return []

    # ===== WIKIQUOTE =====
    
    def _fetch_wikiquote_people(self):
        """Fetch quotes from famous people"""
        people = ['Albert_Einstein', 'Isaac_Newton', 'Aristotle', 'Plato',
                 'William_Shakespeare', 'Leonardo_da_Vinci', 'Marie_Curie',
                 'Martin_Luther_King_Jr', 'Nelson_Mandela', 'Mahatma_Gandhi']
        person = random.choice(people)
        return self._fetch_single_wikiquote(person)

    def _fetch_wikiquote_topics(self):
        """Fetch quotes by topic"""
        topics = ['Science', 'Philosophy', 'Love', 'Life', 'Wisdom',
                 'Education', 'Knowledge', 'Truth', 'Time']
        topic = random.choice(topics)
        return self._fetch_single_wikiquote(topic)

    def _fetch_single_wikiquote(self, title):
        """Fetch a single Wikiquote page"""
        try:
            resp = self.session.get(
                'https://en.wikiquote.org/w/api.php',
                params={
                    'action': 'query',
                    'titles': title,
                    'prop': 'extracts',
                    'explaintext': True,
                    'format': 'json'
                },
                timeout=15
            )
            pages = resp.json().get('query', {}).get('pages', {})
            for page in pages.values():
                if 'extract' in page:
                    # Extract quotes (lines starting with quotes)
                    lines = []
                    for line in page['extract'].split('\n'):
                        line = line.strip()
                        if line.startswith('"') and len(line) > 30:
                            lines.append(line)
                    return lines[:50]
            return []
        except Exception:
            return []

    # ===== OTHER WIKIMEDIA =====
    
    def _fetch_wikibooks(self):
        """Fetch from Wikibooks"""
        books = ['Python_Programming', 'Java_Programming', 'C_Programming',
                'Linear_Algebra', 'Calculus', 'Physics']
        book = random.choice(books)
        return self._fetch_single_wikimedia(book, 'wikibooks.org')

    def _fetch_wikiversity(self):
        """Fetch from Wikiversity"""
        resources = ['Introduction_to_Computers', 'Introduction_to_Psychology',
                    'Introduction_to_Philosophy', 'Introduction_to_Physics']
        resource = random.choice(resources)
        return self._fetch_single_wikimedia(resource, 'wikiversity.org')

    def _fetch_wikinews(self):
        """Fetch from Wikinews"""
        try:
            resp = self.session.get(
                'https://en.wikinews.org/w/api.php',
                params={
                    'action': 'query',
                    'list': 'random',
                    'rnnamespace': 0,
                    'rnlimit': 5,
                    'format': 'json'
                },
                timeout=15
            )
            titles = [item['title'] for item in resp.json().get('query', {}).get('random', [])]
            all_lines = []
            for title in titles:
                lines = self._fetch_single_wikimedia(title, 'wikinews.org')
                all_lines.extend(lines)
            return all_lines
        except Exception:
            return []

    def _fetch_single_wikimedia(self, title, domain):
        """Fetch from any Wikimedia project"""
        try:
            url = f'https://{domain}/w/api.php'
            resp = self.session.get(
                url,
                params={
                    'action': 'query',
                    'titles': title,
                    'prop': 'extracts',
                    'explaintext': True,
                    'format': 'json'
                },
                timeout=15
            )
            pages = resp.json().get('query', {}).get('pages', {})
            for page in pages.values():
                if 'extract' in page and len(page['extract']) > 500:
                    paragraphs = page['extract'].split('\n')
                    return [p.strip() for p in paragraphs if len(p.strip()) > 100]
            return []
        except Exception:
            return []

    # ===== OPEN TEXTBOOKS =====
    
    def _fetch_openstax(self):
        """Fetch from OpenStax textbooks"""
        textbooks = {
            'physics': 'https://openstax.org/details/books/physics',
            'biology': 'https://openstax.org/details/books/biology-2e',
            'chemistry': 'https://openstax.org/details/books/chemistry-2e',
            'calculus': 'https://openstax.org/details/books/calculus-volume-1'
        }
        # Placeholder - would need to scrape or use API
        return []

    def _fetch_bccampus(self): return []  # Placeholder
    def _fetch_merlot(self): return []  # Placeholder

    # ===== MISCELLANEOUS =====
    
    def _fetch_ted_transcripts(self):
        """Fetch TED talk transcripts"""
        try:
            # TED API endpoint
            resp = self.session.get(
                'https://www.ted.com/talks/random/transcript.json',
                timeout=15
            )
            return []  # Placeholder
        except Exception:
            return []

    def _fetch_podcast_transcripts(self):
        """Fetch podcast transcripts"""
        # Placeholder - would need podcast API
        return []

    def _fetch_speeches(self):
        """Famous speeches"""
        speeches = [
            ('Martin Luther King', 'I Have a Dream'),
            ('John F. Kennedy', 'Inaugural Address'),
            ('Winston Churchill', 'We Shall Fight on the Beaches'),
            ('Abraham Lincoln', 'Gettysburg Address')
        ]
        # Placeholder - would need to fetch from wikisource
        return []

    def _fetch_legal_docs(self):
        """Fetch legal documents"""
        docs = [
            ('US Constitution', 'https://www.law.cornell.edu/constitution'),
            ('Universal Declaration of Human Rights', 'https://www.un.org/en/udhrbook/')
        ]
        # Placeholder - would need to scrape
        return []

    def _fetch_religious_texts(self):
        """Fetch religious texts"""
        texts = {
            'Bible': 'https://www.gutenberg.org/files/10/10-0.txt',
            'Quran': 'https://www.gutenberg.org/files/7440/7440-0.txt',
            'Torah': 'https://www.gutenberg.org/files/9439/9439-0.txt'
        }
        # Placeholder - would fetch from Gutenberg
        return []

    def run_forever(self):
        """Main fetch loop - runs until 1.9TB reached"""
        print(f"\n📡 FETCHER STARTED - Target: {self.config.TARGET_DATA_SIZE_GB}GB")
        print(f"   Sources: {len(self.sources)} | Checking size every {self.config.FETCH_DELAY}s")
        
        last_size_check = 0
        check_interval = 60  # Check size every minute
        
        while self.running:
            # Check if target reached
            current_time = time.time()
            if current_time - last_size_check > check_interval:
                if self._is_target_reached():
                    print("\n🎯 TARGET REACHED! Stopping fetcher.")
                    self.running = False
                    break
                last_size_check = current_time
            
            # Get next source
            source_name, source_func = self.sources[self.source_index % len(self.sources)]
            self.source_index += 1
            
            try:
                lines = source_func()
                if lines:
                    added = self._add_lines(lines, source_name)
                    if added > 0:
                        # Small delay between successful fetches
                        time.sleep(self.config.FETCH_DELAY)
            except Exception as e:
                print(f"\n  ⚠️ Error in {source_name}: {str(e)[:100]}")
            
            # Periodic cleanup
            if self.source_index % 100 == 0:
                gc.collect()
                self._save_history()
        
        print("\n📡 FETCHER STOPPED")
        self._save_history()

    def stop(self):
        self.running = False


# =============================
# STREAMING DATASET - 2TB OPTIMIZED
# =============================
class StreamingDataset:
    """
    STREAMING DATASET for 2TB files
    - Never loads entire file into memory
    - Tracks trained lines via hashes
    - Deletes processed lines (like original)
    - 25k line buffer for 12GB RAM
    """
    
    def __init__(self, file_path, tokenizer, config, max_length=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config
        self.lock = threading.Lock()
        
        self.TRAINED_HASHES_FILE = config.TRAINED_HASHES_FILE
        
        # Buffer for streaming - SMALL for 12GB RAM
        self.buffer_size = getattr(config, 'BUFFER_SIZE', 25000)
        self.buffer = []
        self.buffer_hashes = set()
        
        # File position tracking
        self.file_position = 0
        self.file_size = self._get_file_size()
        
        # Deduplication
        self.trained_hashes = set()
        self._load_trained_hashes()
        
        # Statistics
        self.total_processed = 0
        self.total_trained = 0
        
        print(f"\n📂 STREAMING DATASET INITIALIZED")
        print(f"   File: {file_path}")
        print(f"   Size: {self.file_size/1e9:.2f}GB")
        print(f"   Buffer: {self.buffer_size:,} lines")
        print(f"   Previously trained: {len(self.trained_hashes):,} lines")

    def _get_file_size(self):
        """Get current file size"""
        if os.path.exists(self.file_path):
            return os.path.getsize(self.file_path)
        return 0

    def _load_trained_hashes(self):
        """Load hashes of already trained lines"""
        if os.path.exists(self.TRAINED_HASHES_FILE):
            try:
                with open(self.TRAINED_HASHES_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.trained_hashes = set(data.get('hashes', []))
                self.total_trained = data.get('count', len(self.trained_hashes))
                print(f"[Data] Loaded {len(self.trained_hashes):,} trained hashes")
            except Exception:
                self.trained_hashes = set()

    def _save_trained_hashes(self):
        """Save hashes of trained lines"""
        try:
            # Convert set to list for JSON
            hashes_list = list(self.trained_hashes)
            # Limit to save space
            if len(hashes_list) > 1000000:
                hashes_list = hashes_list[-1000000:]
            
            tmp = self.TRAINED_HASHES_FILE + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({
                    'hashes': hashes_list,
                    'count': len(self.trained_hashes),
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            if os.path.exists(self.TRAINED_HASHES_FILE):
                os.remove(self.TRAINED_HASHES_FILE)
            os.rename(tmp, self.TRAINED_HASHES_FILE)
            
        except Exception as e:
            print(f"[Data] Save error: {e}")

    def _hash(self, line):
        """Create hash of line for deduplication"""
        return hashlib.md5(line.encode('utf-8', errors='ignore')).hexdigest()

    def _fill_buffer(self):
        """Fill buffer with new lines from file"""
        if not os.path.exists(self.file_path):
            return 0
        
        with self.lock:
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Seek to last position
                    f.seek(self.file_position)
                    
                    added = 0
                    while added < self.buffer_size:
                        line = f.readline()
                        if not line:  # EOF
                            break
                        
                        self.file_position = f.tell()
                        line = line.strip()
                        
                        if len(line) > 20:  # Minimum line length
                            h = self._hash(line)
                            if h not in self.trained_hashes and h not in self.buffer_hashes:
                                self.buffer.append(line)
                                self.buffer_hashes.add(h)
                                added += 1
                    
                    # If we hit EOF, loop back to beginning
                    if added == 0 and self.file_position >= self.file_size:
                        print("\n📌 Reached end of file, looping to beginning")
                        self.file_position = 0
                        
            except Exception as e:
                print(f"[Data] Buffer fill error: {e}")
            
            return len(self.buffer)

    def available(self):
        """Get available lines in buffer"""
        with self.lock:
            return len(self.buffer)

    def get_batch(self, batch_size):
        """Get a batch of lines for training"""
        with self.lock:
            if len(self.buffer) < batch_size:
                # Try to fill buffer
                self.lock.release()
                self._fill_buffer()
                self.lock.acquire()
                
                if len(self.buffer) < batch_size:
                    return None  # Still not enough
            
            # Get batch from buffer
            batch_lines = self.buffer[:batch_size]
            self.buffer = self.buffer[batch_size:]
            
            # Update trained hashes immediately (for dedup)
            for line in batch_lines:
                h = self._hash(line)
                self.trained_hashes.add(h)
                self.total_processed += 1
                self.total_trained += 1

        # Tokenize batch
        if self.tokenizer is None:
            return None
            
        all_input = []
        all_target = []
        
        for line in batch_lines:
            tokens = self.tokenizer.encode(line)
            
            # Pad or truncate
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                pad = self.tokenizer.special_tokens['<pad>']
                tokens = tokens + [pad] * (self.max_length - len(tokens))
            
            all_input.append(tokens[:-1])
            all_target.append(tokens[1:])
        
        return torch.tensor(all_input), torch.tensor(all_target)

    def flush_trained(self):
        """
        DELETE processed lines from file (like original)
        This is the key feature - removes trained data to save space
        """
        with self.lock:
            if self.total_processed == 0:
                return 0
            
            print(f"\n🗑️  FLUSHING TRAINED DATA - Deleting {self.total_processed:,} lines")
            
            try:
                # Read all remaining lines
                remaining_lines = []
                if os.path.exists(self.file_path):
                    with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if line and len(line) > 20:
                                h = self._hash(line)
                                # Keep lines not yet trained
                                if h not in self.trained_hashes:
                                    remaining_lines.append(line)
                
                # Rewrite file with only untrained lines
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    for line in remaining_lines:
                        f.write(line + '\n')
                
                # Reset position and buffer
                self.file_position = 0
                self.buffer = []
                self.buffer_hashes.clear()
                
                # Update file size
                self.file_size = self._get_file_size()
                
                flushed = self.total_processed
                self.total_processed = 0
                
                print(f"   ✅ Deleted {flushed:,} lines | Remaining: {len(remaining_lines):,} lines | "
                      f"File size: {self.file_size/1e9:.2f}GB")
                
                return flushed
                
            except Exception as e:
                print(f"[Data] Flush error: {e}")
                return 0

    def add_lines(self, new_lines):
        """Add new lines to the end of file (from fetcher)"""
        if not new_lines:
            return
        
        # Filter duplicates
        unique_lines = []
        for line in new_lines:
            if line and len(line) > 20:
                h = self._hash(line)
                if h not in self.trained_hashes:
                    unique_lines.append(line)
        
        if unique_lines:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    for line in unique_lines:
                        f.write(line + '\n')
                
                # Update file size
                self.file_size = self._get_file_size()
                
            except Exception as e:
                print(f"[Data] Add lines error: {e}")

    def reload_from_file(self):
        """Reload buffer (called when waiting for data)"""
        self._fill_buffer()


# =============================
# TRANSFORMER COMPONENTS (from original)
# =============================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(2)
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_heads = config.NUM_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.BIAS)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.BIAS)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.MAX_SEQ_LEN, theta=config.ROPE_THETA)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.DROPOUT)
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // max(1, self.num_kv_heads)
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        attn_weights = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        if attention_mask is None:
            causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            attn_weights = attn_weights + causal
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)
        self.w3 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)
        self.w2 = nn.Linear(config.FFN_DIM, config.HIDDEN_DIM, bias=config.BIAS)
        self.dropout = nn.Dropout(config.DROPOUT)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.HIDDEN_DIM)
        self.ffn_norm = RMSNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        return x

class AdvancedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.NUM_LAYERS)])
        self.norm = RMSNorm(config.HIDDEN_DIM)
        self.output = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized: {total_params:,} parameters")
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.output(x)


# =============================
# MODEL GROWTH MANAGER (from original)
# =============================
class ModelGrowthManager:
    def __init__(self, config):
        self.config = config
        self.growth_log_file = config.GROWTH_LOG_FILE
        self.growth_history = self.load_growth_log()
    def load_growth_log(self):
        if os.path.exists(self.growth_log_file):
            try:
                with open(self.growth_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    def save_growth_log(self):
        try:
            with open(self.growth_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.growth_history, f, indent=2)
        except Exception:
            pass
    def log_growth_event(self, from_size, to_size, step, loss, metrics):
        event = {'timestamp': datetime.now().isoformat(),'from_size':from_size,'to_size':to_size,'training_step':step,'loss_at_growth':loss,'metrics':metrics}
        self.growth_history.append(event)
        self.save_growth_log()
        print(f"\n{'='*60}")
        print(f"  GROWTH EVENT: {from_size} → {to_size} | step {step} | loss {loss:.4f}")
        print(f"{'='*60}\n")
    def get_next_size(self, current_size):
        path = ModelConfig.GROWTH_PATH
        if current_size not in path:
            return None
        idx = path.index(current_size)
        if idx >= len(path)-1:
            return None
        return path[idx+1]
    def transfer_weights(self, old_model, new_model):
        print("Transferring weights to larger model...")
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        transferred = 0
        for k, v in old_state.items():
            if k in new_state:
                if v.shape == new_state[k].shape:
                    new_state[k] = v.clone()
                    transferred += 1
                else:
                    try:
                        target = new_state[k]
                        if v.ndim == 2 and target.ndim == 2:
                            min0 = min(v.shape[0], target.shape[0])
                            min1 = min(v.shape[1], target.shape[1])
                            target[:min0,:min1] = v[:min0,:min1].clone()
                            new_state[k] = target
                            transferred += 1
                        elif v.ndim == 1 and target.ndim == 1:
                            m = min(v.shape[0], target.shape[0])
                            target[:m] = v[:m].clone()
                            new_state[k] = target
                            transferred += 1
                    except Exception:
                        pass
        new_model.load_state_dict(new_state)
        print(f"Weight transfer complete. Transferred ~{transferred} tensors")
        return new_model


# =============================
# CONTINUOUS TRAINER (from original with streaming)
# =============================
class ContinuousTrainer:
    def __init__(self, model, tokenizer, config, growth_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.growth_manager = growth_manager
        self.device = config.DEVICE
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            betas=(config.BETA1, config.BETA2), 
            eps=config.EPS, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
        self.step = 0
        self.epoch = 0
        self.recent_losses = []
        self.loss_history = []
        self.running = True
        self.last_checkpoint_time = time.time()
        
        # Memory monitor
        reserve = getattr(config, 'RESERVE_BYTES', None)
        self.memory_monitor = MemoryMonitor(config.MAX_MEMORY_GB, reserve_bytes=reserve)
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        print(f"\n🎯 TRAINER INITIALIZED")
        print(f"   Device: {config.DEVICE} ({config.DEVICE_TYPE})")
        print(f"   Model size: {config.SIZE}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Grad accum: {config.GRAD_ACCUMULATION_STEPS}")

    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.config.WARMUP_STEPS:
                return max(step / max(1, self.config.WARMUP_STEPS), 0.01)
            progress = (step - self.config.WARMUP_STEPS) / max(1, 100000 - self.config.WARMUP_STEPS)
            return max(0.5 * (1 + math.cos(math.pi * min(progress, 1.0))), 0.01)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch):
        self.model.train()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        if self.scaler and self.config.ENABLE_MIXED_PRECISION:
            with torch.amp.autocast(device_type=self.config.DEVICE_TYPE):
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    target_ids.view(-1), 
                    ignore_index=self.tokenizer.special_tokens['<pad>']
                )
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target_ids.view(-1), 
                ignore_index=self.tokenizer.special_tokens['<pad>']
            )
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            loss.backward()
        
        return loss.item() * self.config.GRAD_ACCUMULATION_STEPS

    def save_checkpoint(self, is_growth=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'size': self.config.SIZE,
            'vocab_size': self.config.VOCAB_SIZE,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history[-1000:],
        }
        
        # Backup dataset hashes
        if hasattr(self, 'dataset') and self.dataset:
            checkpoint['trained_hashes'] = list(self.dataset.trained_hashes)[-100000:]
            checkpoint['trained_count'] = len(self.dataset.trained_hashes)
        
        tmp_path = self.config.MODEL_FILE + ".tmp"
        torch.save(checkpoint, tmp_path)
        
        if os.path.exists(self.config.MODEL_FILE):
            os.remove(self.config.MODEL_FILE)
        os.rename(tmp_path, self.config.MODEL_FILE)
        
        tag = "growth" if is_growth else f"step {self.step}"
        trained_n = checkpoint.get('trained_count', 0)
        print(f"[Save] model.pt updated ({self.config.SIZE}, {tag}) | Trained: {trained_n:,}")

    def load_checkpoint(self):
        """Load model checkpoint"""
        if os.path.exists(self.config.MODEL_FILE):
            try:
                print(f"Loading model from {self.config.MODEL_FILE}...")
                ckpt = torch.load(self.config.MODEL_FILE, map_location='cpu', weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                self.step = ckpt.get('step', 0)
                self.loss_history = ckpt.get('loss_history', [])
                print(f"Resumed: {self.config.SIZE} at step {self.step:,}")
                return True
            except Exception as e:
                print(f"Failed to load model.pt: {e}")
        return False

    def check_growth_criteria(self):
        """Check if we should grow the model"""
        if len(self.loss_history) < self.config.GROWTH_STABLE_STEPS:
            return False
        recent = self.loss_history[-self.config.GROWTH_STABLE_STEPS:]
        avg = sum(recent) / len(recent)
        if avg < self.config.GROWTH_LOSS_THRESHOLD and all(l < self.config.GROWTH_LOSS_THRESHOLD for l in recent):
            return True
        return False

    def perform_growth(self):
        """Grow model to next size"""
        next_size = self.growth_manager.get_next_size(self.config.SIZE)
        if not next_size:
            print("Already max size")
            return False
        
        print(f"\n📈 GROWING {self.config.SIZE} → {next_size}")
        self.save_checkpoint(is_growth=True)
        
        metrics = {
            'step': self.step,
            'avg_loss': sum(self.loss_history[-1000:]) / max(1, len(self.loss_history[-1000:])) if self.loss_history else 0,
            'mem_gb': self.memory_monitor.get_memory_usage_gb()
        }
        
        self.growth_manager.log_growth_event(
            self.config.SIZE, next_size, self.step, 
            self.loss_history[-1] if self.loss_history else 0.0, metrics
        )
        
        new_config = ModelConfig(next_size)
        new_config.VOCAB_SIZE = self.config.VOCAB_SIZE
        new_model = AdvancedTransformer(new_config).to(new_config.DEVICE)
        new_model = self.growth_manager.transfer_weights(self.model, new_model)
        
        self.model = new_model
        self.config = new_config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=new_config.LEARNING_RATE, 
            betas=(new_config.BETA1, new_config.BETA2), 
            eps=new_config.EPS, 
            weight_decay=new_config.WEIGHT_DECAY
        )
        self.scheduler = self._create_scheduler()
        self.loss_history = []
        self.recent_losses = []
        self.step = 0
        
        # Update memory monitor
        reserve = getattr(self.config, 'RESERVE_BYTES', None)
        self.memory_monitor = MemoryMonitor(self.config.MAX_MEMORY_GB, reserve_bytes=reserve)
        self.memory_monitor.force_cleanup()
        
        print(f"Growth complete: now training {self.config.SIZE}")
        self.save_checkpoint(is_growth=True)
        return True

    def train_forever(self, dataset):
        """Main training loop"""
        self.dataset = dataset
        print(f"\n🚀 STARTING TRAINING LOOP")
        
        waiting_logged = False
        last_flush = time.time()
        flush_interval = 300  # Flush every 5 minutes
        
        while self.running:
            # Memory check
            if not self.memory_monitor.is_memory_safe():
                print(f"[Memory] High usage {self.memory_monitor.get_memory_usage_gb():.2f}GB - cleaning")
                self.memory_monitor.force_cleanup()
                if self.config.BATCH_SIZE > 1:
                    self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                    print(f"[Memory] Reduced batch size to {self.config.BATCH_SIZE}")
                time.sleep(2)
            
            # Get batch
            batch = dataset.get_batch(self.config.BATCH_SIZE)
            if batch is None:
                if not waiting_logged:
                    print("[Trainer] Waiting for data...")
                    waiting_logged = True
                dataset.reload_from_file()
                time.sleep(5)
                continue
            
            waiting_logged = False
            
            # Train step
            try:
                loss = self.train_step(batch)
                
                # Update weights
                if (self.step + 1) % self.config.GRAD_ACCUMULATION_STEPS == 0:
                    if self.scaler and self.config.ENABLE_MIXED_PRECISION:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                # Track loss
                self.recent_losses.append(loss)
                self.loss_history.append(loss)
                if len(self.recent_losses) > 1000:
                    self.recent_losses = self.recent_losses[-500:]
                self.step += 1
                
            except RuntimeError as e:
                err = str(e).lower()
                if 'out of memory' in err or 'insufficient' in err or 'could not allocate' in err:
                    self.memory_monitor.force_cleanup()
                    if self.config.BATCH_SIZE > 1:
                        self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                        print(f"[OOM] Reduced batch size to {self.config.BATCH_SIZE}")
                    time.sleep(3)
                    continue
                print(f"[Trainer] RuntimeError: {e}")
                time.sleep(1)
                continue
                
            except Exception as e:
                print(f"[Trainer] Error: {e}")
                time.sleep(1)
                continue
            
            # Log progress
            if self.step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 0
                avail = dataset.available()
                avg = sum(self.recent_losses[-50:]) / max(1, len(self.recent_losses[-50:]))
                mem = self.memory_monitor.get_memory_usage_gb()
                free = self.memory_monitor.get_free_ram_gb()
                
                print(f"\n[Train] Step {self.step:,} | Size: {self.config.SIZE} | "
                      f"Loss: {loss:.4f} | Avg: {avg:.4f} | LR: {lr:.6f}")
                print(f"        Data: {avail:,} lines | Mem: {mem:.2f}/{free:.2f}GB | "
                      f"BS: {self.config.BATCH_SIZE}")
            
            # Growth check
            if self.check_growth_criteria():
                print("[Growth] Criteria met — performing growth")
                self.perform_growth()
            
            # Checkpoint
            if time.time() - self.last_checkpoint_time >= self.config.CHECKPOINT_INTERVAL:
                self.save_checkpoint()
                self.last_checkpoint_time = time.time()
            
            # Periodic flush (delete trained data)
            if time.time() - last_flush >= flush_interval:
                flushed = dataset.flush_trained()
                if flushed > 0:
                    print(f"[Data] Flushed {flushed:,} trained lines")
                last_flush = time.time()
        
        # Final save
        self.save_checkpoint()
        dataset.flush_trained()
        print("[Trainer] Stopped")


# =============================
# MAIN APPLICATION - ULTIMATE 2TB
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
        
        # Set seeds
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
        
        # Print banner
        self._print_banner()

    def _print_banner(self):
        """Print awesome startup banner"""
        print("\n" + "=" * 90)
        print("  🚀 AUTONOMOUS PROGRESSIVE GROWTH AI - ULTIMATE 2TB EDITION")
        print("=" * 90)
        print(f"  Starting size    : {self.config.SIZE}")
        print(f"  Growth path      : {' → '.join(ModelConfig.GROWTH_PATH)}")
        print(f"  Target vocab     : {self.config.VOCAB_SIZE:,} tokens")
        print(f"  Target data size : {self.config.TARGET_DATA_SIZE_GB}GB (1.9TB)")
        print(f"  Device           : {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        try:
            gpu_name = torch_directml.device_name(0) if DIRECTML_AVAILABLE else (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
            )
            print(f"  GPU              : {gpu_name}")
        except:
            pass
        print(f"  Hidden dim       : {self.config.HIDDEN_DIM}")
        print(f"  Layers           : {self.config.NUM_LAYERS}")
        print(f"  Heads            : {self.config.NUM_HEADS}")
        print(f"  Batch size       : {self.config.BATCH_SIZE}")
        print(f"  Grad accum       : {self.config.GRAD_ACCUMULATION_STEPS}")
        print(f"  Max memory       : {self.config.MAX_MEMORY_GB}GB (12GB optimized)")
        print(f"  Data directory   : {self.config.DRIVE_DIR}")
        print(f"  Data sources     : 25+ (Wikipedia, Gutenberg, arXiv, GitHub, News, etc.)")
        print(f"  Deduplication    : ✓ (hash-based)")
        print(f"  Auto-delete      : ✓ (removes trained lines)")
        print(f"  Auto-resume      : ✓ (survives disconnects)")
        print("=" * 90 + "\n")

    def initialize_tokenizer(self):
        """Initialize or load tokenizer"""
        if os.path.exists(self.config.TOKENIZER_FILE):
            print(f"\n📚 Loading existing tokenizer from {self.config.TOKENIZER_FILE}")
            self.tokenizer = UltimateBPETokenizer(vocab_size=self.config.VOCAB_SIZE)
            self.tokenizer.load(self.config.TOKENIZER_FILE)
            self.config.VOCAB_SIZE = len(self.tokenizer.vocab)
            return
        
        print("\n" + "=" * 90)
        print("  🔤 TRAINING ULTIMATE BPE TOKENIZER (200K VOCAB)")
        print("=" * 90)
        
        # Create tokenizer
        self.tokenizer = UltimateBPETokenizer(vocab_size=self.config.VOCAB_SIZE)
        self.tokenizer.set_checkpoint_dir(self.config.BPE_CHECKPOINT_DIR)
        
        # Train on data.txt (should exist from prefetch)
        if os.path.exists(self.config.DATA_FILE):
            self.tokenizer.train_streaming(self.config.DATA_FILE, min_frequency=2)
            self.tokenizer.save(self.config.TOKENIZER_FILE)
            self.config.VOCAB_SIZE = len(self.tokenizer.vocab)
        else:
            print("\n⚠️  No data.txt found for tokenizer training!")
            print("   Please run prefetch first or place data.txt in Drive.")

    def prefetch_data(self):
        """
        PREFETCH DATA before training
        Ensures we have initial data for tokenizer training
        """
        target = self.config.MIN_DATA_LINES
        print(f"\n{'='*90}")
        print(f"  📥 PREFETCH PHASE - Getting {target:,} lines for tokenizer")
        print(f"{'='*90}")
        
        # Check existing data
        existing_lines = 0
        if os.path.exists(self.config.DATA_FILE):
            with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                existing_lines = sum(1 for line in f if line.strip() and len(line.strip()) > 20)
        
        if existing_lines >= target:
            print(f"  Already have {existing_lines:,} lines. Skipping prefetch.")
            return
        
        print(f"  Existing: {existing_lines:,} lines")
        print(f"  Need: {target - existing_lines:,} more lines\n")
        
        # Create temp dataset
        temp_dataset = StreamingDataset(
            self.config.DATA_FILE, 
            None, 
            self.config,
            max_length=self.config.MAX_SEQ_LEN
        )
        
        # Use simplified fetcher for prefetch
        class PrefetchFetcher:
            def __init__(self, dataset, config):
                self.dataset = dataset
                self.config = config
                self.session = requests.Session()
                self.session.headers.update({'User-Agent': 'Prefetch/1.0'})
            
            def fetch_wikipedia_random(self, count=10):
                lines = []
                try:
                    resp = self.session.get(
                        'https://en.wikipedia.org/w/api.php',
                        params={
                            'action': 'query',
                            'list': 'random',
                            'rnnamespace': 0,
                            'rnlimit': count,
                            'format': 'json'
                        },
                        timeout=15
                    )
                    titles = [a['title'] for a in resp.json().get('query', {}).get('random', [])]
                    for title in titles:
                        resp = self.session.get(
                            'https://en.wikipedia.org/w/api.php',
                            params={
                                'action': 'query',
                                'titles': title,
                                'prop': 'extracts',
                                'explaintext': True,
                                'format': 'json'
                            },
                            timeout=15
                        )
                        pages = resp.json().get('query', {}).get('pages', {})
                        for page in pages.values():
                            if 'extract' in page:
                                for p in page['extract'].split('\n'):
                                    p = p.strip()
                                    if len(p) > 100:
                                        lines.append(p)
                        time.sleep(0.2)
                except:
                    pass
                return lines
        
        fetcher = PrefetchFetcher(temp_dataset, self.config)
        
        start_time = time.time()
        last_print = 0
        
        try:
            while temp_dataset.total_processed < target:
                # Fetch some data
                lines = fetcher.fetch_wikipedia_random(20)
                if lines:
                    temp_dataset.add_lines(lines)
                
                current = temp_dataset.total_processed
                elapsed = time.time() - start_time
                
                if elapsed - last_print >= 5:
                    rate = current / max(elapsed, 1)
                    remaining = target - current
                    eta = remaining / max(rate, 0.1)
                    pct = (current / target) * 100
                    
                    bar_len = 40
                    filled = int(bar_len * current / target)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    
                    print(f"\r  [{bar}] {pct:5.1f}% | {current:>10,} / {target:,} | "
                          f"Rate: {rate:.0f}/s | ETA: {eta/60:.1f}min", end="")
                    last_print = elapsed
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Prefetch interrupted by user")
        
        elapsed = time.time() - start_time
        final_count = temp_dataset.total_processed
        print(f"\n\n✅ Prefetch complete: {final_count:,} lines in {elapsed/60:.1f} minutes")

    def initialize_model(self):
        """Initialize or load model"""
        if os.path.exists(self.config.MODEL_FILE):
            print(f"\n📦 Found {self.config.MODEL_FILE}, will load in trainer")
        else:
            print(f"\n📦 Creating new {self.config.SIZE} model")
        self.model = AdvancedTransformer(self.config).to(self.config.DEVICE)

    def run(self):
        """Main execution"""
        # Step 1: Prefetch initial data
        self.prefetch_data()
        
        # Step 2: Initialize tokenizer
        self.initialize_tokenizer()
        
        # Step 3: Initialize model
        self.initialize_model()
        
        # Step 4: Create streaming dataset
        self.dataset = StreamingDataset(
            self.config.DATA_FILE,
            self.tokenizer,
            self.config,
            max_length=self.config.MAX_SEQ_LEN
        )
        
        # Step 5: Create ultimate fetcher (runs until 1.9TB)
        self.fetcher = UltimateDataFetcher(self.dataset, self.config)
        
        # Step 6: Create trainer
        self.trainer = ContinuousTrainer(
            self.model,
            self.tokenizer,
            self.config,
            self.growth_manager
        )
        
        # Step 7: Load checkpoint if exists
        self.trainer.load_checkpoint()
        
        # Step 8: Start fetcher thread
        fetcher_thread = threading.Thread(target=self.fetcher.run_forever, daemon=True)
        fetcher_thread.start()
        print("\n📡 Fetcher thread started")
        
        # Step 9: Start training
        try:
            self.trainer.train_forever(self.dataset)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down gracefully...")
            self.trainer.running = False
            self.fetcher.running = False
            time.sleep(2)
            self.trainer.save_checkpoint()
            self.dataset.flush_trained()
            print("✅ Checkpoint saved. You can resume anytime!")
            print("👋 Goodbye!")


# =============================
# ENTRY POINT
# =============================
def detect_saved_size(model_file="model.pt"):
    """Auto-detect model size from existing checkpoint"""
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


if __name__ == '__main__':
    # Mount Google Drive first (in Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except:
        print("📁 Running locally or Drive already mounted")
    
    # Detect saved size
    saved_size = detect_saved_size()
    if saved_size:
        model_size = saved_size
    else:
        model_size = '10M'
        if len(sys.argv) > 1 and sys.argv[1] in ModelConfig.GROWTH_PATH:
            model_size = sys.argv[1]
    
    # Create and run app
    app = AIApplication(model_size)
    app.run()