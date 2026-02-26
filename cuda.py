#!/usr/bin/env python3
"""
Autonomous Progressive Growth AI Training System
ADVANCED TRAINING-ONLY VERSION - merged improvements from ai.py

Features:
- Continuously fetches data from Wikipedia & many knowledge domains
- Progressive Growth: 10M → 50M → 100M → 200M → 350M → 500M
- Smart checkpointing: Never overwrites, timestamped saves
- Persistent trained-hashes to avoid re-training same data
- Memory optimized: dynamic batching, memory monitor
- Robust DataFetcher with fetched history and expanded category coverage (1000+ generated categories)
- Advanced weight transfer during growth

This file updates the original train.py to include the stronger training pipeline from ai.py while keeping it training-only (no chat server).
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

# Optional DirectML support (if available)
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
# CONFIGURATION
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

        self.VOCAB_SIZE = 32000
        self.MAX_SEQ_LEN = 512
        self.DROPOUT = 0.1
        self.BIAS = False
        self.ROPE_THETA = 10000.0

        # baseline preset batch & grad accum
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
        self.MODEL_FILE = "model.pt"
        self.GROWTH_LOG_FILE = "growth_log.json"

        # checkpoint pruning: keep this many most recent checkpoints
        self.CHECKPOINT_INTERVAL = 300
        self.CHECKPOINT_MAX_KEEP = 3
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 3000
        self.FETCH_BATCH = 10
        self.FETCH_DELAY = 5
        self.PREFETCH_TARGET_LINES = 2_000_000

        self.GROWTH_LOSS_THRESHOLD = 3.0
        self.GROWTH_STABLE_STEPS = 1000

        # Device selection: DirectML > CUDA > CPU
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

        # Aggressive memory targeting: use most of system RAM (95%) for MAX_MEMORY_GB
        if PSUTIL_AVAILABLE:
            total_bytes = psutil.virtual_memory().total
            total_gb = total_bytes / (1024 ** 3)
            # target 95% of system RAM
            self.MAX_MEMORY_GB = round(total_gb * 0.95, 1)
            # reserve ~5% of system RAM (in bytes) for OS and safety; used by MemoryMonitor
            self.RESERVE_BYTES = int(total_bytes * 0.05)
            # auto-scale batch size conservatively based on RAM: every 4GB lets us multiply batch
            scale = max(1, int(total_gb // 4))
            # cap batch size to avoid runaway values
            self.BATCH_SIZE = min(self.BATCH_SIZE * scale, 64)
        else:
            self.MAX_MEMORY_GB = 5.6
            self.RESERVE_BYTES = 64 * 1024 * 1024

        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")
# =============================
# TOKENIZER (ULTIMATE BPE - WITH FULL UNICODE SUPPORT)
# =============================
class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3,'<sep>':4,'<cls>':5,'<mask>':6}
        self.vocab = {}
        self.merges = {}
        self.cache = {}
        self.inverse_vocab = {}
        
        # Advanced features
        self.byte_fallback = True
        self.normalization = 'nfc'
        self.max_word_length = 100
        self.min_frequency = 2
        self.unicode_range = (0x0000, 0x10FFFF)  # Full Unicode range (emojis too!)
        
        # Unicode tracking
        self.unicode_stats = {
            'total_chars_found': 0,
            'unicode_chars_found': 0,
            'unique_unicode': set(),
            'corrupted_lines': 0
        }
        
        # Performance optimizations
        self.cache_size = 10000
        self.progress_interval = 100
        
        # Statistics
        self.stats = {
            'total_lines': 0,
            'unique_words': 0,
            'unicode_words': 0,
            'merge_history': []
        }

    def _normalize(self, text):
        """Advanced Unicode normalization with fallbacks"""
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
                # Verify it's valid UTF-8
                normalized.encode('utf-8')
                return normalized
            except:
                continue
        
        # Ultimate fallback
        return text

    def _detect_encoding(self, text):
        """Detect if text has proper Unicode"""
        if not isinstance(text, str):
            return False
        
        has_unicode = any(ord(c) > 127 for c in text)
        if has_unicode:
            # Track unique Unicode characters
            for c in text:
                if ord(c) > 127:
                    self.unicode_stats['unique_unicode'].add(c)
            self.unicode_stats['unicode_chars_found'] += sum(1 for c in text if ord(c) > 127)
            self.stats['unicode_words'] += 1
        self.unicode_stats['total_chars_found'] += len(text)
        return has_unicode

    def _is_text_corrupted(self, text):
        """Check if text shows signs of encoding corruption"""
        if not isinstance(text, str):
            return True
        
        # Common corruption patterns
        corruption_patterns = [
            '�',  # Unicode replacement character
            '?',  # ASCII fallback for unknown chars (multiple occurrences)
            '\ufffd',  # Unicode replacement character
        ]
        
        # If we see many replacement chars, text is corrupted
        corruption_count = sum(text.count(p) for p in corruption_patterns)
        if corruption_count > len(text) * 0.01:  # More than 1% corrupted
            return True
        
        # Check for mojibake (common encoding errors)
        mojibake_patterns = [
            'Ã©',  # é mis-encoded as Latin-1
            'Ã¼',  # ü mis-encoded
            'Ã±',  # ñ mis-encoded
            'Ã¡',  # á mis-encoded
            'â‚¬',  # € mis-encoded
            'â€™',  # ’ mis-encoded
        ]
        for pattern in mojibake_patterns:
            if pattern in text:
                return True
        
        return False

    def train(self, texts, min_frequency=2, max_merges=None):
        """
        ULTIMATE BPE TRAINING - FULL UNICODE SUPPORT
        Features:
        - Full Unicode support (U+0000 to U+10FFFF)
        - Detects encoding corruption
        - Tracks Unicode statistics
        - Preserves all special characters
        """
        print("=" * 70)
        print("  ULTRA-ADVANCED BPE TOKENIZER TRAINING")
        print("  WITH FULL UNICODE SUPPORT")
        print("=" * 70)
        
        start_time = time.time()
        self.min_frequency = min_frequency
        
        # PHASE 1: Unicode Analysis and Corruption Detection
        print("\n[Phase 1] Analyzing text encoding...")
        unicode_samples = []
        ascii_count = 0
        unicode_count = 0
        total_lines = 0
        corrupted_lines = 0
        
        # Sample first 10000 lines for analysis
        for i, text in enumerate(texts[:10000]):
            if not isinstance(text, str):
                continue
                
            total_lines += 1
            
            # Check for corruption
            if self._is_text_corrupted(text):
                corrupted_lines += 1
                continue
            
            if self._detect_encoding(text):
                unicode_count += 1
                if len(unicode_samples) < 20:
                    # Store sample of Unicode text
                    sample = text[:100]
                    unicode_samples.append(sample)
            else:
                ascii_count += 1
        
        self.unicode_stats['corrupted_lines'] = corrupted_lines
        
        # Print encoding analysis
        print(f"\n  📊 ENCODING ANALYSIS:")
        print(f"  ├─ Total lines sampled: {total_lines:,}")
        print(f"  ├─ ASCII only: {ascii_count:,} lines")
        print(f"  ├─ Unicode detected: {unicode_count:,} lines")
        print(f"  ├─ Corrupted lines: {corrupted_lines:,}")
        print(f"  └─ Unicode characters found: {self.unicode_stats['unicode_chars_found']:,}")
        
        # Show unique Unicode characters found
        unique_unicode = sorted(list(self.unicode_stats['unique_unicode']))[:50]
        if unique_unicode:
            print(f"\n  🔤 UNICODE CHARACTERS FOUND (sample):")
            chunks = []
            for c in unique_unicode:
                try:
                    name = unicodedata.name(c, 'unknown')
                    chunks.append(f"'{c}' (U+{ord(c):04X} {name[:20]})")
                except:
                    chunks.append(f"'{c}' (U+{ord(c):04X})")
            
            # Print in columns
            for i in range(0, len(chunks), 3):
                row = chunks[i:i+3]
                print(f"    {row[0]:<30} {row[1] if len(row)>1 else '':<30} {row[2] if len(row)>2 else ''}")
        
        # Show sample Unicode text
        if unicode_samples:
            print(f"\n  📝 UNICODE TEXT SAMPLES:")
            for sample in unicode_samples[:3]:
                # Show first 50 chars with Unicode visible
                preview = sample[:80]
                print(f"    • {repr(preview)}")
        
        # CRITICAL WARNING if Unicode missing
        if unicode_count == 0:
            print("\n" + "=" * 70)
            print("  ⚠️  ⚠️  ⚠️  CRITICAL WARNING!  ⚠️  ⚠️  ⚠️")
            print("=" * 70)
            print("  NO UNICODE DETECTED IN YOUR DATA!")
            print("\n  Your data.txt contains Unicode but it's being read as ASCII.")
            print("  This will CORRUPT all non-ASCII characters!")
            print("\n  FIX: Add encoding='utf-8' to EVERY file open() call:")
            print("  open(file, 'r', encoding='utf-8')")
            print("\n  Common places to fix:")
            print("  • ConsumingDataset._load_lines()")
            print("  • ConsumingDataset.add_lines()")
            print("  • AIApplication.prefetch_data()")
            print("=" * 70)
            
            # Ask user if they want to continue
            print("\n  Continue anyway? (y/n): ", end='')
            response = input().lower()
            if response != 'y':
                print("  Exiting. Fix encoding and try again.")
                sys.exit(1)
        
        # PHASE 2: Word Frequency Counting (with Unicode preservation)
        print("\n[Phase 2] Counting word frequencies...")
        word_freqs = {}
        total_chars = 0
        valid_lines = 0
        skipped_lines = 0
        unicode_words = 0
        
        for i, text in enumerate(texts):
            # Progress update
            if i % 10000 == 0 and i > 0:
                progress = (i / len(texts)) * 100 if hasattr(texts, '__len__') else 0
                print(f"    Progress: {i:,} lines ({progress:.1f}%) | "
                      f"Unique words: {len(word_freqs):,} | "
                      f"Unicode words: {unicode_words:,}")
            
            # Safety checks
            if text is None:
                skipped_lines += 1
                continue
            
            if not isinstance(text, str):
                try:
                    text = str(text)
                except:
                    skipped_lines += 1
                    continue
            
            if len(text.strip()) == 0:
                skipped_lines += 1
                continue
            
            # Skip corrupted lines
            if self._is_text_corrupted(text):
                skipped_lines += 1
                continue
            
            # Normalize text
            try:
                text = self._normalize(text)
            except:
                text = str(text)
            
            valid_lines += 1
            
            # Process words
            for word in text.strip().split():
                # Check for Unicode
                if any(ord(c) > 127 for c in word):
                    unicode_words += 1
                
                # Limit word length
                if len(word) > self.max_word_length:
                    word = word[:self.max_word_length]
                
                word_freqs[word] = word_freqs.get(word, 0) + 1
                total_chars += len(word)
        
        self.stats['total_lines'] = valid_lines
        self.stats['unique_words'] = len(word_freqs)
        self.stats['unicode_words'] = unicode_words
        
        print(f"\n  📊 DATA STATISTICS:")
        print(f"  ├─ Valid lines: {valid_lines:,}")
        print(f"  ├─ Skipped lines: {skipped_lines:,}")
        print(f"  ├─ Unique words: {len(word_freqs):,}")
        print(f"  ├─ Words with Unicode: {unicode_words:,}")
        print(f"  └─ Total characters: {total_chars:,}")
        
        # Filter rare words
        if min_frequency > 1 and len(word_freqs) > 50000:
            old_size = len(word_freqs)
            word_freqs = {w: f for w, f in word_freqs.items() if f >= min_frequency}
            filtered = old_size - len(word_freqs)
            print(f"\n  🧹 Filtered {filtered:,} rare words (min freq {min_frequency})")
        
        # PHASE 3: Build Unicode-Aware Character Vocabulary
        print("\n[Phase 3] Building Unicode character vocabulary...")
        char_freqs = {}
        vocab = set()
        unicode_set = set()
        
        for word, freq in word_freqs.items():
            for char in word:
                char_freqs[char] = char_freqs.get(char, 0) + freq
                vocab.add(char)
                
                # Track Unicode characters
                if ord(char) > 127:
                    unicode_set.add(char)
            
            vocab.add('</w>')
        
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
        print(f"  ├─ ASCII characters: {len(vocab) - len(unicode_set):,}")
        
        if len(unicode_set) < 100 and unicode_words > 0:
            print(f"\n  ⚠️  WARNING: Found {unicode_words:,} Unicode words but only {len(unicode_set):,} Unicode characters!")
            print("     This suggests encoding corruption. Check file reading.")
        
        # Sort characters by frequency
        sorted_chars = sorted(char_freqs.items(), key=lambda x: -x[1])
        if sorted_chars:
            top_char = sorted_chars[0]
            print(f"\n  📈 Most frequent character: '{top_char[0]}' ({top_char[1]:,} times)")
        
        # Initialize vocabulary with frequency-based ordering
        self.vocab = {}
        
        # First add special tokens with fixed IDs
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        # Then add characters
        next_id = max(self.special_tokens.values()) + 1
        for char in sorted(set(vocab) - set(self.special_tokens.keys())):
            self.vocab[char] = next_id
            next_id += 1
        
        # If vocabulary is too small, add ASCII fallback
        if len(self.vocab) < 1000:
            print("\n  ⚠️  Vocabulary too small - adding ASCII fallback")
            for i in range(32, 127):
                char = chr(i)
                if char not in self.vocab:
                    self.vocab[char] = next_id
                    next_id += 1
        
        # PHASE 4: Initialize splits
        print("\n[Phase 4] Initializing word splits...")
        splits = {}
        for word, freq in word_freqs.items():
            chars = tuple(list(word) + ['</w>'])
            splits[word] = (chars, freq)
        
        # PHASE 5: BPE Merges
        print("\n[Phase 5] Performing BPE merges...")
        
        target_merges = self.vocab_size - len(self.vocab)
        if target_merges <= 0:
            print(f"  ✓ Vocabulary already at target size: {len(self.vocab):,}")
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            elapsed = time.time() - start_time
            print(f"\n  ✓ Tokenizer training complete in {elapsed:.2f} seconds")
            return
        
        max_merges = max_merges or target_merges
        print(f"  Target merges: {max_merges:,} (current vocab: {len(self.vocab):,} → {self.vocab_size:,})")
        print(f"  This will create {max_merges:,} subword tokens from {len(vocab):,} characters")
        
        # Progress tracking
        merge_times = []
        best_scores = []
        
        for merge_idx in range(max_merges):
            merge_start = time.time()
            
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
            
            # Find best pair with advanced scoring
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])
            best_pair_tuple, best_freq = best_pair[0], best_pair[1]
            
            # Advanced scoring: frequency * log(length) * sqrt(unicode_factor)
            merged_token = ''.join(best_pair_tuple)
            unicode_factor = 1.0
            if any(ord(c) > 127 for c in merged_token):
                unicode_factor = 1.5  # Prioritize Unicode merges
            
            score = best_freq * math.log(len(merged_token) + 1) * unicode_factor
            best_scores.append(score)
            
            # Record merge
            self.merges[best_pair_tuple] = merge_idx
            self.stats['merge_history'].append({
                'step': merge_idx,
                'pair': best_pair_tuple,
                'freq': best_freq,
                'score': score
            })
            
            # Update splits
            words_to_update = pair_to_words.get(best_pair_tuple, [])
            new_splits = {}
            
            for word, (chars, freq) in splits.items():
                if word in words_to_update:
                    new_chars = []
                    i = 0
                    while i < len(chars):
                        if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair_tuple:
                            new_chars.append(merged_token)
                            i += 2
                        else:
                            new_chars.append(chars[i])
                            i += 1
                    new_splits[word] = (tuple(new_chars), freq)
                else:
                    new_splits[word] = (chars, freq)
            
            splits = new_splits
            
            # Add to vocabulary
            if merged_token not in self.vocab:
                self.vocab[merged_token] = next_id
                next_id += 1
            
            # Progress tracking
            merge_time = time.time() - merge_start
            merge_times.append(merge_time)
            avg_time = sum(merge_times[-100:]) / max(1, len(merge_times[-100:]))
            
            if (merge_idx + 1) % self.progress_interval == 0:
                elapsed = time.time() - start_time
                remaining = (max_merges - merge_idx - 1) * avg_time
                
                # Progress bar
                pct = (merge_idx + 1) / max_merges * 100
                bar_len = 50
                filled = int(bar_len * (merge_idx + 1) / max_merges)
                bar = '█' * filled + '░' * (bar_len - filled)
                
                # Show Unicode info in progress
                unicode_token = any(ord(c) > 127 for c in merged_token)
                unicode_marker = "✓" if unicode_token else " "
                
                print(f"\r    [{bar}] {pct:5.1f}% | "
                      f"Merges: {merge_idx+1:,}/{max_merges:,} | "
                      f"Vocab: {len(self.vocab):,} | "
                      f"Best: '{best_pair_tuple[0]}' + '{best_pair_tuple[1]}' → '{merged_token}' {unicode_marker}| "
                      f"Freq: {best_freq:,} | "
                      f"ETA: {remaining/60:.1f}min", end="")
            
            # Early stopping conditions
            if len(self.vocab) >= self.vocab_size:
                print(f"\n\n  ✓ Reached target vocabulary size: {len(self.vocab):,}")
                break
            
            if best_freq < 5 and merge_idx > self.vocab_size * 0.5:
                print(f"\n\n  ✓ Early stopping: pair frequency too low ({best_freq})")
                break
        
        print()  # New line after progress
        
        # PHASE 6: Build inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # PHASE 7: Final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("  TOKENIZER TRAINING COMPLETE")
        print("=" * 70)
        print(f"  ✓ Final vocabulary size: {len(self.vocab):,}")
        print(f"  ✓ Total merges performed: {len(self.merges):,}")
        print(f"  ✓ Training time: {elapsed/60:.2f} minutes")
        
        # Vocabulary quality metrics
        token_lengths = [len(t) for t in self.vocab.keys() if t not in self.special_tokens]
        if token_lengths:
            print(f"\n  📊 VOCABULARY STATISTICS:")
            print(f"  ├─ Avg token length: {sum(token_lengths)/len(token_lengths):.2f} chars")
            print(f"  ├─ Min token length: {min(token_lengths)} chars")
            print(f"  ├─ Max token length: {max(token_lengths)} chars")
        
        # Unicode statistics
        unicode_tokens = [t for t in self.vocab.keys() if any(ord(c) > 127 for c in t)]
        if unicode_tokens:
            print(f"\n  🌍 UNICODE SUPPORT:")
            print(f"  ├─ Unicode tokens: {len(unicode_tokens):,}")
            print(f"  ├─ Unicode ratio: {len(unicode_tokens)/len(self.vocab)*100:.1f}%")
            
            # Show sample Unicode tokens
            sample_unicode = sorted(unicode_tokens)[:20]
            print(f"  └─ Examples: {', '.join(sample_unicode)}")
        else:
            print(f"\n  ⚠️  NO UNICODE TOKENS CREATED!")
            print("     Your data is being read as ASCII. Add encoding='utf-8' to all file opens!")
        
        print("=" * 70)

    def encode(self, text, add_special_tokens=True):
        """Fast encoding with Unicode preservation"""
        if text is None:
            text = ""
        
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                text = ""
        
        # Check for corruption
        if self._is_text_corrupted(text):
            # Log but continue
            pass
        
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
            
            # Convert to IDs - use 'replace' to preserve characters
            for char in chars:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                elif self.byte_fallback:
                    # Byte fallback for unknown chars - use 'replace' to preserve characters
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
        """Advanced decoding with Unicode reconstruction"""
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
        # Convert set to list for JSON serialization
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
                'version': '3.0-unicode-ultimate',
                'unicode_support': 'full'
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n  ✓ Tokenizer saved to {path}")
        print(f"    - Vocabulary: {len(self.vocab):,} tokens")
        print(f"    - Unicode tokens: {sum(1 for t in self.vocab if any(ord(c)>127 for c in t)):,}")
        print(f"    - Merges: {len(self.merges):,}")

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
        
        # Load Unicode stats if available
        unicode_stats = data.get('unicode_stats', {})
        self.unicode_stats['total_chars_found'] = unicode_stats.get('total_chars_found', 0)
        self.unicode_stats['unicode_chars_found'] = unicode_stats.get('unicode_chars_found', 0)
        self.unicode_stats['corrupted_lines'] = unicode_stats.get('corrupted_lines', 0)
        
        print(f"\n  ✓ Tokenizer loaded from {path}")
        print(f"    - Vocabulary: {len(self.vocab):,} tokens")
        print(f"    - Unicode tokens: {sum(1 for t in self.vocab if any(ord(c)>127 for c in t)):,}")
        print(f"    - Merges: {len(self.merges):,}")
        
        # Verify Unicode support
        unicode_test = "café über αβγ 今日は 日本語"
        encoded = self.encode(unicode_test, add_special_tokens=False)
        decoded = self.decode(encoded, skip_special_tokens=True)
        print(f"\n  🔤 Unicode test: '{unicode_test}'")
        print(f"  🔄 Decoded:      '{decoded}'")
        
        if decoded != unicode_test and all(ord(c) < 128 for c in decoded):
            print("  ⚠️  WARNING: Unicode may not be properly preserved!")
            print("     Check file encoding when loading training data.")
# MEMORY MONITOR
# =============================
class MemoryMonitor:
    # small default reserve; ModelConfig will pass a more accurate reserve_bytes when available
    RESERVE_BYTES = 64 * 1024 * 1024
    def __init__(self, max_memory_gb=5.0, reserve_bytes=None):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        if reserve_bytes is not None:
            self.RESERVE_BYTES = reserve_bytes
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
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
    def is_memory_safe(self):
        if not self.process:
            return True
        if PSUTIL_AVAILABLE:
            # consider memory safe if available is greater than the configured reserve
            return psutil.virtual_memory().available > self.RESERVE_BYTES
        return self.get_memory_usage() < self.max_memory_bytes * 0.9
    def can_boost(self):
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available > self.RESERVE_BYTES
        return False
    def force_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if DIRECTML_AVAILABLE:
            gc.collect()

# =============================
# MODEL GROWTH MANAGER
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
        # Smart transfer: copy matching tensors, slice partially where needed
        transferred = 0
        for k, v in old_state.items():
            if k in new_state:
                if v.shape == new_state[k].shape:
                    new_state[k] = v.clone()
                    transferred += 1
                else:
                    # try to copy partial slices
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
# TRANSFORMER (lightweight)
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
# DATASET (persistent hashes)
# =============================
class ConsumingDataset:
    TRAINED_HASHES_FILE = "trained_data.json"
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lock = threading.Lock()
        self.lines = []
        self.pending_hashes = set()
        self.trained_hashes = set()
        self.consumed_count = 0
        self._load_trained_hashes()
        self._load_lines()
    def _load_trained_hashes(self):
        if os.path.exists(self.TRAINED_HASHES_FILE):
            try:
                with open(self.TRAINED_HASHES_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.trained_hashes = set(data.get('hashes', []))
                print(f"[Data] Loaded {len(self.trained_hashes)} trained data hashes")
            except Exception:
                self.trained_hashes = set()
    def _save_trained_hashes(self):
        try:
            tmp = self.TRAINED_HASHES_FILE + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'hashes': list(self.trained_hashes), 'count': len(self.trained_hashes)}, f)
            if os.path.exists(self.TRAINED_HASHES_FILE):
                os.remove(self.TRAINED_HASHES_FILE)
            os.rename(tmp, self.TRAINED_HASHES_FILE)
        except Exception:
            pass
    def _hash(self, line):
        # Use 'strict' to fail on encoding errors rather than silently dropping characters
        # This ensures Greek letters (αβγ), accents (éüñ), em-dashes (—), and smart quotes ("") are preserved
        return hashlib.md5(line.encode('utf-8', errors='strict')).hexdigest()
    def _load_lines(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                pass
            self.lines = []
            return
        # Use errors='strict' to preserve all Unicode characters including
        # Greek letters (αβγ), accents (éüñ), em-dashes (—), and smart quotes ("")
        with open(self.file_path, 'r', encoding='utf-8', errors='strict') as f:
            raw = [l.strip() for l in f if l.strip() and len(l.strip()) > 20]
        with self.lock:
            self.lines = []
            self.pending_hashes = set()
            for line in raw:
                h = self._hash(line)
                if h not in self.trained_hashes and h not in self.pending_hashes:
                    self.pending_hashes.add(h)
                    self.lines.append(line)
            self.consumed_count = 0
        if len(raw) != len(self.lines):
            self._rewrite_file()
    def _rewrite_file(self):
        try:
            tmp = self.file_path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                for line in self.lines:
                    f.write(line + '\n')
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            os.rename(tmp, self.file_path)
        except Exception:
            pass
    def available(self):
        with self.lock:
            return len(self.lines) - self.consumed_count
    def total_lines(self):
        with self.lock:
            return len(self.lines)
    def get_batch(self, batch_size):
        with self.lock:
            if self.consumed_count >= len(self.lines):
                return None
            batch_lines = self.lines[self.consumed_count:self.consumed_count + batch_size]
            self.consumed_count += len(batch_lines)
        if self.tokenizer is None:
            return None
        all_input = []
        all_target = []
        for line in batch_lines:
            tokens = self.tokenizer.encode(line)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                pad = self.tokenizer.special_tokens['<pad>']
                tokens = tokens + [pad] * (self.max_length - len(tokens))
            all_input.append(tokens[:-1])
            all_target.append(tokens[1:])
        return torch.tensor(all_input), torch.tensor(all_target)
    def flush_trained(self):
        with self.lock:
            if self.consumed_count == 0:
                return 0
            trained_lines = self.lines[:self.consumed_count]
            for line in trained_lines:
                self.trained_hashes.add(self._hash(line))
            self.lines = self.lines[self.consumed_count:]
            flushed = self.consumed_count
            self.consumed_count = 0
        self._save_trained_hashes()
        self._rewrite_file()
        return flushed
    def flush_consumed(self):
        return 0
    def add_lines(self, new_lines):
        clean = [l.strip() for l in new_lines if l.strip() and len(l.strip()) > 20]
        if not clean:
            return
        unique = []
        with self.lock:
            for line in clean:
                h = self._hash(line)
                if h not in self.trained_hashes and h not in self.pending_hashes:
                    self.pending_hashes.add(h)
                    self.lines.append(line)
                    unique.append(line)
        if unique:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    for line in unique:
                        f.write(line + '\n')
            except Exception:
                pass
    def reload_from_file(self):
        self._load_lines()
# =============================
# DATA FETCHER (expanded categories)
# =============================
class DataFetcher:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.fetched_count = 0
        self.session = requests.Session()
        self.session.headers.update({'User-Agent':'AutonomousAITrainer/1.0'})
        self.fetched_urls = set()
        self.history_file = Path("fetched_history.json")
        if self.history_file.exists():
            try:
                data = json.load(self.history_file.open('r', encoding='utf-8'))
                if isinstance(data, dict):
                    self.fetched_urls = set(data.get('fetched_urls', []))
                    self.category_idx = data.get('category_idx', 0)
                    self.fetched_count = data.get('fetched_count', 0)
                    print(f"[Fetcher] Resumed — {len(self.fetched_urls)} sources already fetched, category_idx={self.category_idx}")
                else:
                    self.fetched_urls = set(data)
            except Exception:
                self.fetched_urls = set()
        # base categories (combined from original files)
        base = [
            "Science", "Technology", "Mathematics", "Physics", "Chemistry", "Biology",
            "Computer_science", "Engineering", "Medicine", "Astronomy", "Geology",
            "History", "Philosophy", "Psychology", "Sociology", "Economics",
            "Politics", "Geography", "Literature", "Art", "Music"
        ]
        # programmatically expand to 1000+ categories by creating sensible variants
        self.WIKI_CATEGORIES = self._expand_categories(base, target=1100)
        self.category_idx = 0
        self.gutenberg_ids = list(range(1, 400))  # larger set for diversity
        self.vital_levels = [1,2,3,4,5]
        self.search_queries = [
            "fundamental physics concepts", "cellular biology mechanisms",
            "historical civilizations development", "modern technology advances",
            "mathematical theorems proofs", "climate science research",
            "economic theories models", "psychological phenomena studies",
            "philosophical arguments debates", "linguistic structures analysis"
        ]

    def _expand_categories(self, base_list, target=1100):
        out = []
        seen = set()
        suffixes = ["", "_history", "_overview", "_theory", "_applications", "_biography", "_concepts", "_research", "_introduction"]
        for b in base_list:
            for s in suffixes:
                cand = (b + s).replace(' ', '_')
                if cand not in seen:
                    out.append(cand)
                    seen.add(cand)
        # Generate subtopics by combining words
        words = sorted({w for b in base_list for w in re.split(r'[_\s]+', b)})
        i = 0
        while len(out) < target:
            a = random.choice(words)
            b = random.choice(words)
            cand = f"{a}_{b}"
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            i += 1
            if i > target * 5:
                break
        return out

    def _save_history(self):
        try:
            data = {
                'fetched_urls': list(self.fetched_urls),
                'category_idx': self.category_idx,
                'fetched_count': self.fetched_count,
            }
            tmp = str(self.history_file) + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            if self.history_file.exists():
                os.remove(self.history_file)
            os.rename(tmp, str(self.history_file))
        except Exception:
            pass

    def _clean_text(self, text):
        text = re.sub(r'\[\d+]', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'==+\s*See also\s*==+', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*References\s*==+', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*External links\s*==+', '', text, flags=re.DOTALL)
        text = re.sub(r'==+[^=]+=+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        lines = []
        for para in text.split('\n'):
            para = para.strip()
            if len(para) > 50 and sum(c.isalpha() for c in para) > len(para) * 0.4:
                if len(para) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    chunk = ''
                    for s in sentences:
                        if len(chunk) + len(s) > 400:
                            if len(chunk) > 50:
                                lines.append(chunk.strip())
                            chunk = s
                        else:
                            chunk += ' ' + s if chunk else s
                    if len(chunk) > 50:
                        lines.append(chunk.strip())
                else:
                    lines.append(para)
        return lines

    def _fetch_wiki_article(self, title):
        if title in self.fetched_urls:
            return []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','titles':title,'prop':'extracts','explaintext':True,'format':'json'}, timeout=15)
            pages = resp.json().get('query', {}).get('pages', {})
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 200:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(title)
                    return lines
        except Exception:
            pass
        return []

    def fetch_wikipedia_random(self, count=10):
        all_lines = []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','list':'random','rnnamespace':0,'rnlimit':count,'format':'json'}, timeout=15)
            titles = [a['title'] for a in resp.json().get('query', {}).get('random', [])]
            for title in titles:
                lines = self._fetch_wiki_article(title)
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.2)
        except Exception as e:
            print(f"[Fetcher] Wikipedia random error: {e}")
        return all_lines

    def fetch_wikipedia_category(self):
        category = self.WIKI_CATEGORIES[self.category_idx % len(self.WIKI_CATEGORIES)]
        self.category_idx += 1
        all_lines = []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','list':'categorymembers','cmtitle':f'Category:{category}','cmlimit':15,'cmtype':'page','format':'json'}, timeout=15)
            members = resp.json().get('query', {}).get('categorymembers', [])
            for member in members[:10]:
                lines = self._fetch_wiki_article(member['title'])
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.2)
        except Exception as e:
            print(f"[Fetcher] Category '{category}' error: {e}")
        return all_lines, category

    def fetch_wikipedia_vital(self):
        topic = random.choice(["Universe","Earth","Life","Human","Science","Mathematics"])
        lines = self._fetch_wiki_article(topic)
        return lines, topic

    def fetch_wikipedia_search(self):
        query = random.choice(self.search_queries)
        all_lines = []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','list':'search','srsearch':query,'srlimit':5,'format':'json'}, timeout=15)
            results = resp.json().get('query', {}).get('search', [])
            for r in results:
                lines = self._fetch_wiki_article(r['title'])
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.2)
        except Exception as e:
            print(f"[Fetcher] Wiki search '{query}' error: {e}")
        return all_lines, query

    def fetch_gutenberg(self):
        # choose a random id from the list; attempt two URL patterns
        book_id = random.choice(self.gutenberg_ids)
        url1 = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
        url2 = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
        for url in (url1, url2):
            if url in self.fetched_urls:
                continue
            try:
                resp = self.session.get(url, timeout=20)
                if resp.status_code == 200:
                    text = resp.content.decode('utf-8', errors='ignore')
                    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 30]
                    self.fetched_urls.add(url)
                    return lines[:200]
            except Exception:
                pass
        return []

    def fetch_wikisource(self):
        # sample a primary document
        doc_list = ['United_States_Constitution','Declaration_of_Independence_(United_States)','Magna_Carta']
        doc = random.choice(doc_list)
        key = f'wikisource:{doc}'
        if key in self.fetched_urls:
            return [], doc
        try:
            resp = self.session.get('https://en.wikisource.org/w/api.php', params={'action':'query','titles':doc,'prop':'extracts','explaintext':True,'format':'json'}, timeout=15)
            pages = resp.json().get('query', {}).get('pages', {})
            for pid,page in pages.items():
                if 'extract' in page and len(page['extract']) > 100:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines, doc
        except Exception as e:
            print(f"[Fetcher] Wikisource {doc} error: {e}")
        return [], doc

    def fetch_wikiquote(self):
        people = ['Albert_Einstein','Isaac_Newton','Aristotle','Plato']
        person = random.choice(people)
        key = f'wikiquote:{person}'
        if key in self.fetched_urls:
            return []
        try:
            resp = self.session.get('https://en.wikiquote.org/w/api.php', params={'action':'query','titles':person,'prop':'extracts','explaintext':True,'format':'json'}, timeout=15)
            pages = resp.json().get('query', {}).get('pages', {})
            for pid,page in pages.items():
                if 'extract' in page and len(page['extract']) > 50:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines
        except Exception:
            pass
        return []

    def fetch_open_textbook(self):
        urls = [
            'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        ]
        url = random.choice(urls)
        if url in self.fetched_urls:
            return []
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            text = resp.content.decode('utf-8', errors='ignore')
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip())>30]
            self.fetched_urls.add(url)
            return lines
        except Exception:
            return []

    def run_forever(self):
        sources = [
            ('Wikipedia (random)', lambda: self.fetch_wikipedia_random(self.config.FETCH_BATCH)),
            ('Wikipedia (category)', lambda: self.fetch_wikipedia_category()[0]),
            ('Wikipedia (vital)', lambda: self.fetch_wikipedia_vital()[0]),
            ('Wikipedia (search)', lambda: self.fetch_wikipedia_search()[0]),
            ('Project Gutenberg', lambda: self.fetch_gutenberg()),
            ('Wikisource', lambda: self.fetch_wikisource()[0]),
            ('Wikiquote', lambda: self.fetch_wikiquote()),
            ('Open textbook', lambda: self.fetch_open_textbook()),
        ]
        idx = 0
        print(f"[Fetcher] Started — {len(self.WIKI_CATEGORIES)} categories available")
        while self.running:
            src_name, src_fn = sources[idx % len(sources)]
            try:
                lines = src_fn()
                if lines:
                    self.dataset.add_lines(lines)
                    self.fetched_count += len(lines)
                    print(f"[Fetcher] {src_name}: +{len(lines)} lines (total fetched: {self.fetched_count}) | available: {self.dataset.available()}")
                    self._save_history()
            except Exception as e:
                print(f"[Fetcher] {src_name} error: {e}")
            idx += 1
            time.sleep(self.config.FETCH_DELAY)
    def stop(self):
        self.running = False

# =============================
# CONTINUOUS TRAINER (focused)
# =============================
class ContinuousTrainer:
    def __init__(self, model, tokenizer, config, growth_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.growth_manager = growth_manager
        self.device = config.DEVICE
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2), eps=config.EPS, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
        self.step = 0
        self.epoch = 0
        self.recent_losses = []
        self.loss_history = []
        self.running = True
        self.last_checkpoint_time = time.time()
        # pass reserve_bytes from config to MemoryMonitor so it uses the configured small safety reserve
        reserve = getattr(config, 'RESERVE_BYTES', None)
        self.memory_monitor = MemoryMonitor(config.MAX_MEMORY_GB, reserve_bytes=reserve)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
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
            with torch.amp.autocast():
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=self.tokenizer.special_tokens['<pad>'])
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=self.tokenizer.special_tokens['<pad>'])
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            loss.backward()
        return loss.item() * self.config.GRAD_ACCUMULATION_STEPS
    def save_checkpoint(self, is_growth=False):
        """Save model to single model.pt file (overwrites previous)"""
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
        # Backup dedup state inside model.pt
        if hasattr(self, 'dataset') and self.dataset:
            checkpoint['trained_hashes'] = list(self.dataset.trained_hashes)
            checkpoint['trained_count'] = len(self.dataset.trained_hashes)
        if hasattr(self, 'fetcher') and self.fetcher:
            checkpoint['fetcher_state'] = {
                'fetched_urls': list(self.fetcher.fetched_urls),
                'category_idx': self.fetcher.category_idx,
                'fetched_count': self.fetcher.fetched_count,
            }
        tmp_path = self.config.MODEL_FILE + ".tmp"
        torch.save(checkpoint, tmp_path)
        if os.path.exists(self.config.MODEL_FILE):
            os.remove(self.config.MODEL_FILE)
        os.rename(tmp_path, self.config.MODEL_FILE)
        tag = "growth" if is_growth else f"step {self.step}"
        trained_n = checkpoint.get('trained_count', 0)
        fetched_n = len(checkpoint.get('fetcher_state', {}).get('fetched_urls', []))
        print(f"[Save] model.pt updated ({self.config.SIZE}, {tag}) | Dedup: {trained_n:,} trained, {fetched_n:,} fetched")
    def load_checkpoint(self):
        """Load from single model.pt file"""
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
        if len(self.loss_history) < self.config.GROWTH_STABLE_STEPS:
            return False
        recent = self.loss_history[-self.config.GROWTH_STABLE_STEPS:]
        avg = sum(recent) / len(recent)
        if avg < self.config.GROWTH_LOSS_THRESHOLD and all(l < self.config.GROWTH_LOSS_THRESHOLD for l in recent):
            return True
        return False
    def perform_growth(self):
        next_size = self.growth_manager.get_next_size(self.config.SIZE)
        if not next_size:
            print("Already max size")
            return False
        print(f"Growing {self.config.SIZE} → {next_size}")
        self.save_checkpoint(is_growth=True)
        metrics = {'step':self.step,'avg_loss': sum(self.loss_history[-1000:]) / max(1,len(self.loss_history[-1000:])) if self.loss_history else 0,'mem_gb':self.memory_monitor.get_memory_usage_gb()}
        self.growth_manager.log_growth_event(self.config.SIZE, next_size, self.step, self.loss_history[-1] if self.loss_history else 0.0, metrics)
        new_config = ModelConfig(next_size)
        new_config.VOCAB_SIZE = self.config.VOCAB_SIZE
        new_model = AdvancedTransformer(new_config).to(new_config.DEVICE)
        new_model = self.growth_manager.transfer_weights(self.model, new_model)
        self.model = new_model
        self.config = new_config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=new_config.LEARNING_RATE, betas=(new_config.BETA1, new_config.BETA2), eps=new_config.EPS, weight_decay=new_config.WEIGHT_DECAY)
        self.scheduler = self._create_scheduler()
        self.loss_history = []
        self.recent_losses = []
        self.step = 0
        # after growth ensure memory monitor uses new config values
        reserve = getattr(self.config, 'RESERVE_BYTES', None)
        self.memory_monitor = MemoryMonitor(self.config.MAX_MEMORY_GB, reserve_bytes=reserve)
        self.memory_monitor.force_cleanup()
        print(f"Growth complete: now training {self.config.SIZE}")
        self.save_checkpoint(is_growth=True)
        return True
    def train_forever(self, dataset):
        self.dataset = dataset
        print(f"[Trainer] Starting training on {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        waiting_logged = False
        while self.running:
            if not self.memory_monitor.is_memory_safe():
                print(f"[Memory] High usage {self.memory_monitor.get_memory_usage_gb():.2f}GB - cleaning")
                self.memory_monitor.force_cleanup()
                if self.config.BATCH_SIZE > 1:
                    self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                    print(f"[Memory] Reduced batch size to {self.config.BATCH_SIZE}")
                time.sleep(2)
            batch = dataset.get_batch(self.config.BATCH_SIZE)
            if batch is None:
                if not waiting_logged:
                    print("[Trainer] Waiting for data...")
                    waiting_logged = True
                dataset.reload_from_file()
                time.sleep(5)
                continue
            waiting_logged = False
            try:
                loss = self.train_step(batch)
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
            if self.step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 0
                avail = dataset.available()
                avg = sum(self.recent_losses[-50:]) / max(1, len(self.recent_losses[-50:]))
                mem = self.memory_monitor.get_memory_usage_gb()
                print(f"[Train] Step {self.step} | Size: {self.config.SIZE} | Loss: {loss:.4f} | Avg: {avg:.4f} | LR: {lr:.6f} | Data: {avail} | Mem: {mem:.2f}GB | BS: {self.config.BATCH_SIZE}")
            if self.check_growth_criteria():
                print("[Growth] Criteria met — performing growth")
                self.perform_growth()
            if time.time() - self.last_checkpoint_time >= self.config.CHECKPOINT_INTERVAL:
                self.save_checkpoint()
                flushed = dataset.flush_trained()
                if flushed > 0:
                    print(f"[Data] Flushed {flushed} trained lines")
                self.last_checkpoint_time = time.time()
        # final save
        self.save_checkpoint()
        dataset.flush_trained()
        print("[Trainer] Stopped")

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
        print(f"\n{'='*60}")
        print("  AUTONOMOUS PROGRESSIVE GROWTH AI - TRAINING ONLY")
        print(f"{'='*60}")
        print(f"  Starting size : {model_size}")
        print(f"  Growth path   : {' → '.join(ModelConfig.GROWTH_PATH)}")
        try:
            gpu_name = torch_directml.device_name(0) if DIRECTML_AVAILABLE else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
        except Exception:
            gpu_name = 'N/A'
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
        print(f"  Data sources  : Wikipedia (expanded categories), Gutenberg, Wikisource, Wikiquote")
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
            with open(self.config.DATA_FILE, 'w', encoding='utf-8') as f:
                pass
        temp_dataset = ConsumingDataset(self.config.DATA_FILE, None, self.config.MAX_SEQ_LEN)
        temp_fetcher = DataFetcher(temp_dataset, self.config)
        bootstrap = [
            lambda: temp_fetcher.fetch_wikipedia_random(20),
            lambda: temp_fetcher.fetch_gutenberg(),
            lambda: temp_fetcher.fetch_wikipedia_category()[0],
            lambda: temp_fetcher.fetch_wikipedia_vital()[0],
            lambda: temp_fetcher.fetch_wikipedia_search()[0],
            lambda: temp_fetcher.fetch_wikisource()[0],
            lambda: temp_fetcher.fetch_wikiquote(),
            lambda: temp_fetcher.fetch_open_textbook(),
        ]
        idx = 0
        while temp_dataset.total_lines() < self.config.MIN_DATA_LINES:
            print(f"  Data: {temp_dataset.total_lines()} / {self.config.MIN_DATA_LINES} | fetching...")
            try:
                lines = bootstrap[idx % len(bootstrap)]()
                if lines:
                    temp_dataset.add_lines(lines)
            except Exception as e:
                print(f"Bootstrap error: {e}")
            idx += 1
            time.sleep(1)
        texts = []
        with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if line.strip():
                    texts.append(line.strip())
            print(f"  Loaded {i+1:,} lines for tokenizer training")
        
        self.tokenizer = BPETokenizer(vocab_size=self.config.VOCAB_SIZE)
        self.tokenizer.train(texts)
        self.tokenizer.save(self.config.TOKENIZER_FILE)
        self.config.VOCAB_SIZE = len(self.tokenizer.vocab)
        self.fetched_urls_bootstrap = temp_fetcher.fetched_urls
    def initialize_model(self):
        if os.path.exists(self.config.MODEL_FILE):
            print(f"Found {self.config.MODEL_FILE}, will load {self.config.SIZE} model in trainer")
        else:
            print(f"No model.pt found, creating new {self.config.SIZE} model")
        self.model = AdvancedTransformer(self.config).to(self.config.DEVICE)
    def prefetch_data(self):
        """Aggressively fetch ~2M lines of training data before training begins."""
        target = self.config.PREFETCH_TARGET_LINES
        print(f"\n{'='*60}")
        print(f"  PREFETCH PHASE")
        print(f"  Target: {target:,} lines of training data")
        print(f"  Sources: Wikipedia, Gutenberg, Wikisource, Wikiquote")
        print(f"  Press Ctrl+C to skip and start training early")
        print(f"{'='*60}\n")

        start_time = time.time()

        existing_lines = 0
        if os.path.exists(self.config.DATA_FILE):
            with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                existing_lines = sum(1 for line in f if line.strip() and len(line.strip()) > 20)

        if existing_lines >= target:
            print(f"  Already have {existing_lines:,} lines (>= {target:,}). Skipping prefetch.")
            return

        print(f"  Existing data: {existing_lines:,} lines")
        print(f"  Need to fetch: {target - existing_lines:,} more lines\n")

        temp_dataset = ConsumingDataset(self.config.DATA_FILE, None, self.config.MAX_SEQ_LEN)
        fetcher = DataFetcher(temp_dataset, self.config)
        if hasattr(self, 'fetched_urls_bootstrap'):
            fetcher.fetched_urls.update(self.fetched_urls_bootstrap)

        sources = [
            ("Wikipedia (random)", lambda: fetcher.fetch_wikipedia_random(20)),
            ("Wikipedia (category)", lambda: fetcher.fetch_wikipedia_category()[0]),
            ("Wikipedia (vital)", lambda: fetcher.fetch_wikipedia_vital()[0]),
            ("Wikipedia (search)", lambda: fetcher.fetch_wikipedia_search()[0]),
            ("Project Gutenberg", lambda: fetcher.fetch_gutenberg()),
            ("Wikisource", lambda: fetcher.fetch_wikisource()[0]),
            ("Wikiquote", lambda: fetcher.fetch_wikiquote()),
            ("Open textbook", lambda: fetcher.fetch_open_textbook()),
        ]

        round_idx = 0
        last_print = 0

        try:
            while temp_dataset.total_lines() < target:
                src_name, src_fn = sources[round_idx % len(sources)]
                try:
                    lines = src_fn()
                    if lines:
                        temp_dataset.add_lines(lines)
                        fetcher.fetched_count += len(lines)
                except Exception as e:
                    print(f"  Fetch error ({src_name}): {e}")
                    round_idx += 1
                    time.sleep(1)
                    continue

                current = temp_dataset.total_lines()
                elapsed = time.time() - start_time

                if elapsed - last_print >= 5 or round_idx % 10 == 0:
                    rate = current / max(elapsed, 1)
                    remaining = target - current
                    eta = remaining / max(rate, 0.1)
                    pct = current / target * 100
                    bar_len = 30
                    filled = int(bar_len * current / target)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    print(f"  [{bar}] {pct:5.1f}% | {current:>10,} / {target:,} | "
                          f"Rate: {rate:.0f}/s | ETA: {eta/60:.0f}min")
                    last_print = elapsed

                round_idx += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n  Prefetch interrupted by user. Got {temp_dataset.total_lines():,} lines.")

        elapsed = time.time() - start_time
        final_count = temp_dataset.total_lines()
        print(f"\n  Prefetch complete: {final_count:,} lines in {elapsed/60:.1f} minutes")
        print(f"  Average rate: {final_count/max(elapsed,1):.0f} lines/second\n")

        self.fetched_urls_bootstrap = fetcher.fetched_urls
        fetcher._save_history()

    def run(self):
        self.initialize_tokenizer()
        # Prefetch phase: fetch ~2M lines before training
        self.prefetch_data()
        self.initialize_model()
        self.dataset = ConsumingDataset(self.config.DATA_FILE, self.tokenizer, self.config.MAX_SEQ_LEN)
        self.fetcher = DataFetcher(self.dataset, self.config)
        if hasattr(self, 'fetched_urls_bootstrap'):
            self.fetcher.fetched_urls.update(self.fetched_urls_bootstrap)
        # Restore dedup state from model.pt if JSON files are incomplete/missing
        if os.path.exists(self.config.MODEL_FILE):
            try:
                ckpt = torch.load(self.config.MODEL_FILE, map_location='cpu', weights_only=False)
                backup_hashes = set(ckpt.get('trained_hashes', []))
                if backup_hashes and len(backup_hashes) > len(self.dataset.trained_hashes):
                    added = len(backup_hashes) - len(self.dataset.trained_hashes)
                    self.dataset.trained_hashes.update(backup_hashes)
                    self.dataset._save_trained_hashes()
                    self.dataset._load_lines()
                    print(f"[Dedup] Restored {added:,} trained hashes from model.pt backup")
                fs = ckpt.get('fetcher_state', {})
                if fs:
                    backup_urls = set(fs.get('fetched_urls', []))
                    if len(backup_urls) > len(self.fetcher.fetched_urls):
                        added = len(backup_urls) - len(self.fetcher.fetched_urls)
                        self.fetcher.fetched_urls.update(backup_urls)
                        self.fetcher.category_idx = max(self.fetcher.category_idx, fs.get('category_idx', 0))
                        self.fetcher.fetched_count = max(self.fetcher.fetched_count, fs.get('fetched_count', 0))
                        self.fetcher._save_history()
                        print(f"[Dedup] Restored {added:,} fetched URLs from model.pt backup")
                del ckpt
                gc.collect()
            except Exception as e:
                print(f"[Dedup] Could not restore from model.pt: {e}")
        fetcher_thread = threading.Thread(target=self.fetcher.run_forever, daemon=True)
        fetcher_thread.start()
        self.trainer = ContinuousTrainer(self.model, self.tokenizer, self.config, self.growth_manager)
        self.trainer.fetcher = self.fetcher
        self.trainer.load_checkpoint()
        try:
            self.trainer.train_forever(self.dataset)
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
            self.trainer.running = False
            self.fetcher.running = False
            time.sleep(2)
            self.trainer.save_checkpoint()
            self.dataset.flush_trained()
            print("Checkpoint saved. You can resume anytime with: python train.py")
            print("Goodbye!")
# ENTRY POINT
def detect_saved_size(model_file="model.pt"):
    """Auto-detect model size from existing model.pt on restart."""
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
    saved_size = detect_saved_size()
    if saved_size:
        model_size = saved_size
    else:
        model_size = '10M'
        if len(sys.argv) > 1 and sys.argv[1] in ModelConfig.GROWTH_PATH:
            model_size = sys.argv[1]
    
    print(f"\n  Model: {model_size} ({'Resuming' if saved_size else 'Fresh start'})")
    print(f"  Saves to: model.pt (auto-overwrite)\n")
    
    app = AIApplication(model_size)
    app.run()