#!/usr/bin/env python3
"""
===============================================================================
██████╗ ███████╗███████╗██████╗ ███████╗███████╗██████╗ 
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗
██║  ██║█████╗  █████╗  ██████╔╝█████╗  █████╗  ██████╔╝
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██╔══╝  ██╔══██╗
██████╔╝███████╗███████╗██║     ██║     ███████╗██║  ██║
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝     ╚══════╝╚═╝  ╚═╝

   ULTRA-ADVANCED AI TRAINING SYSTEM - ENTERPRISE EDITION
===============================================================================
"""

import os
import sys
import time
import json
import math
import random
import hashlib
import threading
import re
import logging
import gc
import gzip
import glob  
import pickle
import signal
import unicodedata
import queue
import mmap
import struct
import socket
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
from typing import Optional, List, Dict, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager, nullcontext
from functools import wraps, lru_cache
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# Optional but highly recommended imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ Install psutil for better monitoring: pip install psutil")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("⚠️ Install GPUtil for GPU monitoring: pip install gputil")

try:
    from datasets import load_dataset, Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.modules.mha import FlashSelfAttention
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ Flash Attention not available")

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("⚠️ xFormers not available")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️ DeepSpeed not available")

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not available for logging")

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

# Tokenizers import
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️ Install tokenizers: pip install tokenizers")

# =============================================================================
# ENTERPRISE CONFIGURATION WITH AUTO-DETECTION
# =============================================================================

@dataclass
class SystemCapabilities:
    """Auto-detected system capabilities"""
    cpu_count: int = 0
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    cuda_version: str = ""
    cudnn_version: str = ""
    torch_version: str = ""
    python_version: str = ""
    platform: str = ""
    is_colab: bool = False
    is_kaggle: bool = False
    is_paperspace: bool = False
    network_speed_mbps: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    
    def __post_init__(self):
        """Auto-detect all capabilities"""
        self._detect_hardware()
        self._detect_software()
        self._detect_environment()
        self._detect_network()
        self._detect_storage()
    
    def _detect_hardware(self):
        """Detect hardware capabilities"""
        # CPU
        self.cpu_count = os.cpu_count() or 0
        self.cpu_cores = psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else self.cpu_count
        
        # RAM
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            self.ram_total_gb = mem.total / (1024**3)
            self.ram_available_gb = mem.available / (1024**3)
        
        # GPU
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            for i in range(self.gpu_count):
                self.gpu_names.append(torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                self.gpu_memory_gb.append(props.total_memory / (1024**3))
            
            self.cuda_version = torch.version.cuda or ""
            self.cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
    
    def _detect_software(self):
        """Detect software versions"""
        self.torch_version = torch.__version__
        self.python_version = platform.python_version()
    
    def _detect_environment(self):
        """Detect cloud environment"""
        # Google Colab detection
        try:
            import google.colab
            self.is_colab = True
        except ImportError:
            pass
        
        # Kaggle detection
        try:
            import kaggle_secrets
            self.is_kaggle = True
        except ImportError:
            pass
        
        # Paperspace detection
        try:
            import paperspace
            self.is_paperspace = True
        except ImportError:
            pass
        
        self.platform = platform.platform()
    
    def _detect_network(self):
        """Detect network speed (approximate)"""
        try:
            # Test download speed to HuggingFace
            import requests
            start = time.time()
            r = requests.get("https://huggingface.co", timeout=2)
            elapsed = time.time() - start
            # Rough estimate: 1MB download would take similar time
            self.network_speed_mbps = 8 / elapsed if elapsed > 0 else 100
        except:
            self.network_speed_mbps = 100  # Assume 100Mbps
    
    def _detect_storage(self):
        """Detect storage capabilities"""
        try:
            if PSUTIL_AVAILABLE:
                disk = psutil.disk_usage('/content' if self.is_colab else '/')
                self.disk_free_gb = disk.free / (1024**3)
                self.disk_total_gb = disk.total / (1024**3)
        except:
            pass
    
    def get_optimal_batch_size(self, model_size_mb: float = 1000) -> int:
        """Dynamically calculate optimal batch size"""
        if self.gpu_count > 0:
            # GPU-based calculation
            total_gpu_memory = sum(self.gpu_memory_gb)
            # Reserve 20% for overhead
            usable_memory = total_gpu_memory * 0.8 * 1024  # Convert to MB
            
            # Each sample takes roughly model_size_mb * sequence_length/1024
            sample_memory = model_size_mb * 2  # Rough estimate
            batch_size = max(1, int(usable_memory / sample_memory))
            
            # Adjust for multi-GPU
            batch_size = batch_size * self.gpu_count
            
            return min(batch_size, 64)  # Cap at 64 for stability
        else:
            # CPU-based calculation
            usable_memory = self.ram_available_gb * 0.5 * 1024  # 50% of RAM in MB
            sample_memory = model_size_mb * 2
            batch_size = max(1, int(usable_memory / sample_memory))
            return min(batch_size, 16)  # CPU batch size limit
    
    def get_optimal_seq_length(self, min_seq: int = 1024) -> int:
        """Dynamically calculate optimal sequence length"""
        base_seq = min_seq
        
        # Scale with RAM
        if self.ram_total_gb >= 64:
            base_seq = max(base_seq, 8192)
        elif self.ram_total_gb >= 32:
            base_seq = max(base_seq, 4096)
        elif self.ram_total_gb >= 16:
            base_seq = max(base_seq, 2048)
        
        # Scale with GPU memory
        if self.gpu_memory_gb and max(self.gpu_memory_gb) >= 24:
            base_seq = max(base_seq, 4096)
        elif self.gpu_memory_gb and max(self.gpu_memory_gb) >= 16:
            base_seq = max(base_seq, 2048)
        
        # Adjust for flash attention capabilities
        if FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE:
            base_seq = base_seq * 2  # Flash attention handles longer sequences
        
        return base_seq
    
    def get_recommended_precision(self) -> str:
        """Get recommended precision based on hardware"""
        if self.gpu_count == 0:
            return "fp32"
        
        # Check for bfloat16 support
        if torch.cuda.is_bf16_supported():
            return "bf16"
        
        # Check for tensor cores
        if any('V100' in name or 'A100' in name or 'H100' in name for name in self.gpu_names):
            return "fp16"
        
        return "fp32"
    
    def get_optimal_num_workers(self) -> int:
        """Get optimal number of data loading workers"""
        # Leave one core for training
        return max(1, self.cpu_cores - 1)
    
    def get_optimal_gradient_accumulation(self, target_batch_size: int = 32) -> int:
        """Calculate gradient accumulation steps"""
        actual_batch_size = self.get_optimal_batch_size()
        steps = max(1, target_batch_size // actual_batch_size)
        return steps
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def print_summary(self):
        """Print comprehensive system summary"""
        print("\n" + "="*80)
        print("🚀 SYSTEM CAPABILITIES REPORT")
        print("="*80)
        
        print(f"\n📊 HARDWARE:")
        print(f"  • CPU: {self.cpu_cores} cores / {self.cpu_count} threads")
        print(f"  • RAM: {self.ram_total_gb:.1f}GB total, {self.ram_available_gb:.1f}GB available")
        
        if self.gpu_count > 0:
            print(f"\n🎮 GPU:")
            for i, (name, memory) in enumerate(zip(self.gpu_names, self.gpu_memory_gb)):
                print(f"  • GPU {i}: {name} ({memory:.1f}GB)")
            print(f"  • CUDA: {self.cuda_version}")
            print(f"  • cuDNN: {self.cudnn_version}")
        
        print(f"\n💻 SOFTWARE:")
        print(f"  • PyTorch: {self.torch_version}")
        print(f"  • Python: {self.python_version}")
        print(f"  • Platform: {self.platform}")
        
        print(f"\n🌐 ENVIRONMENT:")
        if self.is_colab:
            print("  • Google Colab")
        if self.is_kaggle:
            print("  • Kaggle")
        if self.is_paperspace:
            print("  • Paperspace")
        
        print(f"\n📡 NETWORK:")
        print(f"  • Estimated speed: {self.network_speed_mbps:.1f} Mbps")
        
        print(f"\n💾 STORAGE:")
        print(f"  • Free: {self.disk_free_gb:.1f}GB")
        print(f"  • Total: {self.disk_total_gb:.1f}GB")
        
        print(f"\n⚙️ OPTIMAL CONFIGURATION:")
        print(f"  • Batch size: {self.get_optimal_batch_size()}")
        print(f"  • Sequence length: {self.get_optimal_seq_length()}")
        print(f"  • Precision: {self.get_recommended_precision()}")
        print(f"  • Workers: {self.get_optimal_num_workers()}")
        
        print("\n" + "="*80)


@dataclass
class UltraAdvancedConfig:
    """Enterprise-grade configuration with auto-tuning"""
    
    # Auto-detected capabilities
    caps: SystemCapabilities = field(default_factory=SystemCapabilities)
    
    # Model architecture - dynamically scaled
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = None  # Will be 4 * model_dim if None
    max_seq_length: int = None  # Will be auto-detected
    vocab_size: int = 200000
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embed_dropout: float = 0.1
    
    # Advanced architecture features
    use_flash_attention: bool = FLASH_ATTN_AVAILABLE
    use_xformers: bool = XFORMERS_AVAILABLE
    use_sdpa: bool = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    use_compile: bool = False  # PyTorch 2.0 compile
    use_checkpointing: bool = True  # Gradient checkpointing
    use_memory_efficient_attention: bool = True
    use_alibi: bool = False  # Attention with Linear Biases
    use_rotary: bool = True  # Rotary Position Embeddings
    use_rms_norm: bool = True  # RMSNorm instead of LayerNorm
    use_swiglu: bool = True  # SwiGLU activation
    use_multiquery: bool = False  # Multi-Query Attention
    use_grouped_query: bool = True  # Grouped-Query Attention
    num_kv_heads: int = None  # For GQA, will be num_heads // 4 if None
    
    # Training hyperparameters - auto-tuned
    batch_size: int = None  # Will be auto-detected
    eval_batch_size: int = None
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 32  # Target effective batch size
    
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 1_000_000
    decay_steps: int = None  # Will be max_steps - warmup_steps
    scheduler_type: str = "cosine"  # linear, cosine, onecycle, cosine_with_restarts
    
    # Precision
    precision: str = None  # Will be auto-detected
    enable_amp: bool = True
    enable_tf32: bool = True
    enable_bf16: bool = None  # Will be auto-detected
    
    # Data processing
    num_workers: int = None  # Will be auto-detected
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Paths
    base_dir: str = "/content/drive/MyDrive/ai.train"
    data_dir: str = None
    inbox_dir: str = None
    checkpoint_dir: str = None
    bpe_checkpoint_dir: str = None
    log_dir: str = None
    metrics_dir: str = None
    cache_dir: str = None
    profile_dir: str = None
    
    # Checkpointing
    save_every_steps: int = 1000
    keep_last_n_checkpoints: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    
    # Logging & Monitoring
    log_every_steps: int = 100
    eval_every_steps: int = 1000
    profile_every_steps: int = 10000
    use_wandb: bool = WANDB_AVAILABLE
    wandb_project: str = "ultra-advanced-ai"
    wandb_run_name: str = None
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    use_deepspeed: bool = DEEPSPEED_AVAILABLE
    deepspeed_config: Dict = field(default_factory=dict)
    
    # Memory optimization
    max_memory_usage_gb: float = None  # Will be auto-detected
    empty_cache_freq: int = 100
    gc_freq: int = 1000
    
    # BPE training
    bpe_vocab_size: int = 200000
    bpe_min_frequency: int = 2
    
    def __post_init__(self):
        """Auto-configure based on system capabilities"""
        # Feedforward dimension
        if self.ff_dim is None:
            self.ff_dim = 4 * self.model_dim
        
        # Sequence length
        if self.max_seq_length is None:
            self.max_seq_length = self.caps.get_optimal_seq_length()
        
        # Batch size
        if self.batch_size is None:
            self.batch_size = self.caps.get_optimal_batch_size()
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size * 2
        
        # Gradient accumulation
        if self.gradient_accumulation_steps == 1:
            self.gradient_accumulation_steps = self.caps.get_optimal_gradient_accumulation(
                self.effective_batch_size
            )
        
        # Precision
        if self.precision is None:
            self.precision = self.caps.get_recommended_precision()
        if self.enable_bf16 is None:
            self.enable_bf16 = self.precision == "bf16"
        
        # Workers
        if self.num_workers is None:
            self.num_workers = self.caps.get_optimal_num_workers()
        
        # Decay steps
        if self.decay_steps is None:
            self.decay_steps = self.max_steps - self.warmup_steps
        
        # KV heads for GQA
        if self.use_grouped_query and self.num_kv_heads is None:
            self.num_kv_heads = max(1, self.num_heads // 4)
        
        # Memory limit
        if self.max_memory_usage_gb is None:
            self.max_memory_usage_gb = self.caps.ram_total_gb * 0.8
        
        # Setup directories
        self._setup_directories()
        
        # Enable TF32 for Ampere GPUs
        if self.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _setup_directories(self):
        """Setup all directories"""
        if self.data_dir is None:
            self.data_dir = f"{self.base_dir}/data"
        if self.inbox_dir is None:
            self.inbox_dir = f"{self.base_dir}/inbox"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"{self.base_dir}/checkpoints"
        if self.bpe_checkpoint_dir is None:
            self.bpe_checkpoint_dir = f"{self.base_dir}/bpe_checkpoints"
        if self.log_dir is None:
            self.log_dir = f"{self.base_dir}/logs"
        if self.metrics_dir is None:
            self.metrics_dir = f"{self.base_dir}/metrics"
        if self.cache_dir is None:
            self.cache_dir = f"{self.base_dir}/cache"
        if self.profile_dir is None:
            self.profile_dir = f"{self.base_dir}/profiles"
        
        # Create all directories
        for dir_path in [
            self.data_dir, self.inbox_dir, self.checkpoint_dir, 
            self.bpe_checkpoint_dir, self.log_dir, self.metrics_dir,
            self.cache_dir, self.profile_dir
        ]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_training_args(self) -> Dict:
        """Get training arguments for logging"""
        return {
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_length': self.max_seq_length,
            'vocab_size': self.vocab_size,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'effective_batch_size': self.batch_size * self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_steps': self.max_steps,
            'warmup_steps': self.warmup_steps,
            'precision': self.precision,
        }
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("⚙️ TRAINING CONFIGURATION")
        print("="*80)
        
        print(f"\n🏗️ MODEL:")
        print(f"  • Dimension: {self.model_dim}")
        print(f"  • Heads: {self.num_heads}")
        print(f"  • Layers: {self.num_layers}")
        print(f"  • FFN: {self.ff_dim}")
        print(f"  • Vocab: {self.vocab_size:,}")
        print(f"  • Seq Length: {self.max_seq_length:,}")
        
        print(f"\n🎯 TRAINING:")
        print(f"  • Batch Size: {self.batch_size}")
        print(f"  • Grad Accumulation: {self.gradient_accumulation_steps}")
        print(f"  • Effective Batch: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"  • Learning Rate: {self.learning_rate}")
        print(f"  • Weight Decay: {self.weight_decay}")
        print(f"  • Max Steps: {self.max_steps:,}")
        print(f"  • Warmup Steps: {self.warmup_steps:,}")
        
        print(f"\n💾 PRECISION:")
        print(f"  • Type: {self.precision}")
        print(f"  • AMP: {self.enable_amp}")
        print(f"  • TF32: {self.enable_tf32}")
        
        print(f"\n🚀 OPTIMIZATIONS:")
        print(f"  • Flash Attention: {self.use_flash_attention}")
        print(f"  • xFormers: {self.use_xformers}")
        print(f"  • SDPA: {self.use_sdpa}")
        print(f"  • Checkpointing: {self.use_checkpointing}")
        print(f"  • RMSNorm: {self.use_rms_norm}")
        print(f"  • SwiGLU: {self.use_swiglu}")
        print(f"  • Rotary: {self.use_rotary}")
        
        print("\n" + "="*80)


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

class MemoryManager:
    """Enterprise-grade memory management"""
    
    def __init__(self, config: UltraAdvancedConfig):
        self.config = config
        self.max_memory_gb = config.max_memory_usage_gb
        
        # Tracking
        self.allocations = {}
        self.peak_memory = 0
        self.warnings = []
        
        # Monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._check_memory()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"⚠️ Memory monitor error: {e}")
    
    def _check_memory(self):
        """Check memory usage"""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            
            if used_gb > self.max_memory_gb * 0.9:
                self.warnings.append({
                    'time': time.time(),
                    'message': f"High RAM usage: {used_gb:.1f}GB / {self.max_memory_gb:.1f}GB"
                })
                self.cleanup()
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                if allocated > self.peak_memory:
                    self.peak_memory = allocated
                
                # Check if close to limit
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / (1024**3)
                
                if allocated > total * 0.9:
                    self.warnings.append({
                        'time': time.time(),
                        'message': f"High GPU{i} usage: {allocated:.1f}GB / {total:.1f}GB"
                    })
                    torch.cuda.empty_cache()
    
    @contextmanager
    def track_allocation(self, name: str):
        """Track memory allocation for a block"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated() / (1024**3)
        else:
            start_mem = 0
        
        start_time = time.time()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated() / (1024**3)
            else:
                end_mem = 0
            
            elapsed = time.time() - start_time
            allocated = end_mem - start_mem
            
            self.allocations[name] = {
                'allocated_gb': allocated,
                'time_seconds': elapsed,
                'timestamp': time.time()
            }
    
    def cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_ram_usage(self) -> float:
        """Get current RAM usage in GB"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().used / (1024**3)
        return 0.0
    
    def get_gpu_usage(self) -> float:
        """Get current GPU usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        stats = {
            'ram_current_gb': self.get_ram_usage(),
            'ram_peak_gb': self.max_memory_gb,
            'gpu_current_gb': self.get_gpu_usage(),
            'gpu_peak_gb': self.peak_memory,
            'gpu_cached_gb': torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0,
            'allocations': len(self.allocations),
            'warnings': len(self.warnings),
            'last_warnings': self.warnings[-5:] if self.warnings else []
        }
        
        return stats
    
    def print_stats(self):
        """Print memory statistics"""
        stats = self.get_stats()
        
        print("\n📊 MEMORY STATISTICS:")
        print(f"  RAM: {stats['ram_current_gb']:.2f}GB (peak: {stats['ram_peak_gb']:.2f}GB)")
        if stats['gpu_current_gb'] > 0:
            print(f"  GPU: {stats['gpu_current_gb']:.2f}GB (peak: {stats['gpu_peak_gb']:.2f}GB)")
            print(f"  GPU Cached: {stats['gpu_cached_gb']:.2f}GB")
        print(f"  Allocations tracked: {stats['allocations']}")
        
        if stats['warnings'] > 0:
            print(f"  ⚠️ Warnings: {stats['warnings']}")
            for w in stats['last_warnings']:
                print(f"    • {w['message']}")
    
    def shutdown(self):
        """Clean shutdown"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        self.cleanup()


# =============================================================================
# ADVANCED DISTRIBUTED TRAINING
# =============================================================================

class DistributedManager:
    """Enterprise-grade distributed training manager"""
    
    def __init__(self, config: UltraAdvancedConfig):
        self.config = config
        self.is_distributed = False
        self.is_main = True
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        self._setup_distributed()
    
    def _setup_distributed(self):
        """Setup distributed training if available"""
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.is_distributed = self.world_size > 1
            self.is_main = self.rank == 0
            
            if self.is_distributed:
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
                torch.cuda.set_device(self.local_rank)
                
                print(f"🌐 Distributed training initialized: rank {self.rank}/{self.world_size-1}")
    
    def barrier(self):
        """Synchronize all processes"""
        if self.is_distributed:
            torch.distributed.barrier()
    
    def reduce_metric(self, metric: float, op='mean') -> float:
        """Reduce metric across all processes"""
        if not self.is_distributed:
            return metric
        
        tensor = torch.tensor(metric).cuda()
        if op == 'mean':
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            tensor /= self.world_size
        elif op == 'sum':
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        elif op == 'max':
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        
        return tensor.item()
    
    def get_sampler(self, dataset) -> Optional[DistributedSampler]:
        """Get distributed sampler"""
        if self.is_distributed:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        return None
    
    def shutdown(self):
        """Clean shutdown"""
        if self.is_distributed:
            torch.distributed.destroy_process_group()


# =============================================================================
# ENTERPRISE LOGGING & MONITORING
# =============================================================================

class EnterpriseLogger:
    """Comprehensive logging with multiple backends"""
    
    def __init__(self, config: UltraAdvancedConfig, name: str = "training"):
        self.config = config
        self.name = name
        self.log_file = f"{config.log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.metrics_file = f"{config.metrics_dir}/{name}_metrics.jsonl"
        self.profile_file = f"{config.profile_dir}/{name}_profile.json"
        
        self.metrics = []
        self.events = []
        self.start_time = time.time()
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup wandb
        if config.use_wandb and WANDB_AVAILABLE and self.is_main():
            self._setup_wandb()
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        import logging
        
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        self.logger.addHandler(fh)
        
        # Console handler (only for main process)
        if self.is_main():
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s'
            ))
            self.logger.addHandler(ch)
    
    def _setup_wandb(self):
        """Setup Weights & Biases"""
        try:
            run_name = self.config.wandb_run_name or f"{self.name}_{datetime.now():%Y%m%d_%H%M%S}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=self.config.get_training_args(),
                dir=self.config.log_dir,
                save_code=True
            )
            self.logger.info(f"📊 WandB initialized: {run_name}")
        except Exception as e:
            self.logger.warning(f"⚠️ WandB initialization failed: {e}")
    
    def is_main(self) -> bool:
        """Check if this is the main process"""
        if 'RANK' in os.environ:
            return int(os.environ['RANK']) == 0
        return True
    
    def log(self, level: str, message: str):
        """Log a message"""
        self.logger.log(getattr(logging, level.upper(), logging.INFO), message)
        
        self.events.append({
            'time': time.time(),
            'level': level,
            'message': message
        })
    
    def info(self, message: str):
        self.log('info', message)
    
    def warning(self, message: str):
        self.log('warning', f"⚠️ {message}")
    
    def error(self, message: str):
        self.log('error', f"❌ {message}")
    
    def metric(self, name: str, value: float, step: int = None):
        """Log a metric"""
        metric = {
            'time': time.time(),
            'name': name,
            'value': value,
            'step': step if step is not None else len(self.metrics)
        }
        self.metrics.append(metric)
        
        # Write to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric) + '\n')
        
        # Log to wandb
        if self.config.use_wandb and WANDB_AVAILABLE and self.is_main():
            wandb.log({name: value}, step=step)
    
    def metrics_dict(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.metric(name, value, step)
    
    def get_metrics(self, name: str, last_n: int = None) -> List[float]:
        """Get metric history"""
        filtered = [m for m in self.metrics if m['name'] == name]
        if last_n:
            filtered = filtered[-last_n:]
        return [m['value'] for m in filtered]
    
    def get_average(self, name: str, last_n: int = 100) -> float:
        """Get average of recent metrics"""
        values = self.get_metrics(name, last_n)
        return sum(values) / len(values) if values else 0.0
    
    def save_profile(self, profile_data: Dict):
        """Save profiling data"""
        with open(self.profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        elapsed = time.time() - self.start_time
        
        summary = {
            'name': self.name,
            'duration_seconds': elapsed,
            'duration_hours': elapsed / 3600,
            'metrics_logged': len(self.metrics),
            'events_logged': len(self.events),
            'log_file': self.log_file,
            'metrics_file': self.metrics_file,
        }
        
        # Add latest metrics
        latest = {}
        for metric in reversed(self.metrics[-100:]):
            if metric['name'] not in latest:
                latest[metric['name']] = metric['value']
        summary['latest_metrics'] = latest
        
        return summary
    
    def print_summary(self):
        """Print training summary"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY")
        print("="*80)
        print(f"  Duration: {summary['duration_hours']:.2f} hours")
        print(f"  Metrics logged: {summary['metrics_logged']}")
        print(f"  Events logged: {summary['events_logged']}")
        print("\n  Latest metrics:")
        for name, value in summary['latest_metrics'].items():
            print(f"    • {name}: {value:.4f}")
        print("="*80)
    
    def shutdown(self):
        """Clean shutdown"""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        self.print_summary()


# =============================================================================
# DATA PROCESSING & MANAGEMENT
# =============================================================================

class AdvancedMultiFileDataManager:
    """Advanced data manager for processing multiple files"""
    
    def __init__(self, config: UltraAdvancedConfig):
        self.config = config
        self.processed_files = set()
        self.file_hashes = {}
        
        # Load processed files log
        self.log_file = f"{config.data_dir}/processed_files.json"
        self._load_log()
    
    def _load_log(self):
        """Load processed files log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('files', []))
                    self.file_hashes = data.get('hashes', {})
            except:
                pass
    
    def _save_log(self):
        """Save processed files log"""
        with open(self.log_file, 'w') as f:
            json.dump({
                'files': list(self.processed_files),
                'hashes': self.file_hashes
            }, f, indent=2)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get file hash"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def process_files(self, max_files: int = None) -> int:
        """Process files from inbox"""
        inbox_files = glob.glob(f"{self.config.inbox_dir}/*")
        processed_count = 0
        
        for filepath in inbox_files:
            if max_files and processed_count >= max_files:
                break
            
            # Check if already processed
            file_hash = self._get_file_hash(filepath)
            if filepath in self.processed_files or file_hash in self.file_hashes.values():
                continue
            
            try:
                # Read and process file
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Clean and split into lines
                lines = content.split('\n')
                clean_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
                
                # Save to training data
                output_file = f"{self.config.data_dir}/training_data_{len(self.processed_files):06d}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(clean_lines))
                
                # Mark as processed
                self.processed_files.add(filepath)
                self.file_hashes[filepath] = file_hash
                processed_count += 1
                
                print(f"✅ Processed: {os.path.basename(filepath)} -> {len(clean_lines)} lines")
                
            except Exception as e:
                print(f"⚠️ Error processing {filepath}: {e}")
        
        # Save log
        self._save_log()
        
        return processed_count


class StreamingDataset(IterableDataset):
    """Enterprise-grade streaming dataset with prefetching"""
    
    def __init__(
        self,
        file_pattern: str,
        tokenizer,
        config: UltraAdvancedConfig,
        shuffle: bool = True,
        infinite: bool = False
    ):
        self.file_pattern = file_pattern
        self.tokenizer = tokenizer
        self.config = config
        self.shuffle = shuffle
        self.infinite = infinite
        
        self.files = []
        self._refresh_file_list()
        
        # Statistics
        self.lines_read = 0
        self.bytes_read = 0
        self.start_time = time.time()
    
    def _refresh_file_list(self):
        """Refresh the list of files"""
        self.files = glob.glob(self.file_pattern)
        if self.shuffle:
            random.shuffle(self.files)
    
    def __iter__(self):
        """Iterate through the dataset"""
        while True:
            for file in self.files:
                if not os.path.exists(file):
                    continue
                
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if self.shuffle:
                        random.shuffle(lines)
                    
                    for line in lines:
                        line = line.strip()
                        if not line or len(line) < 10:
                            continue
                        
                        # Tokenize
                        encoding = self.tokenizer.encode(line)
                        input_ids = encoding.ids
                        
                        # Truncate if needed
                        if len(input_ids) > self.config.max_seq_length:
                            input_ids = input_ids[:self.config.max_seq_length]
                        
                        # Create sample
                        sample = {
                            'input_ids': input_ids,
                            'attention_mask': [1] * len(input_ids),
                            'text': line
                        }
                        
                        self.lines_read += 1
                        self.bytes_read += len(line.encode('utf-8'))
                        
                        yield sample
                
                except Exception as e:
                    print(f"⚠️ Error reading {file}: {e}")
            
            if not self.infinite:
                break
            
            # Refresh file list for infinite mode
            self._refresh_file_list()
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        elapsed = time.time() - self.start_time
        rate = self.lines_read / elapsed if elapsed > 0 else 0
        
        return {
            'lines_read': self.lines_read,
            'bytes_read_gb': self.bytes_read / 1e9,
            'lines_per_second': rate,
            'files_available': len(self.files)
        }


def collate_fn(batch):
    """Collate function for data loader"""
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_mask = []
    
    for item in batch:
        ids = item['input_ids']
        mask = item['attention_mask']
        
        # Pad
        padding_length = max_len - len(ids)
        ids = ids + [0] * padding_length
        mask = mask + [0] * padding_length
        
        input_ids.append(ids)
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    
    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


class MultiHeadAttention(nn.Module):
    """Advanced multi-head attention with multiple backends"""
    
    def __init__(self, config: UltraAdvancedConfig):
        super().__init__()
        self.config = config
        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Grouped-query attention
        if config.use_grouped_query:
            self.num_kv_heads = config.num_kv_heads or config.num_heads
        else:
            self.num_kv_heads = config.num_heads
        
        # Projections
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        
        # Rotary embeddings
        if config.use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_length)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        # Backend selection
        self.use_flash = config.use_flash_attention and FLASH_ATTN_AVAILABLE
        self.use_xformers = config.use_xformers and XFORMERS_AVAILABLE
        self.use_sdpa = config.use_sdpa
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if self.config.use_rotary:
            cos, sin = self.rotary(seq_len)
            q, k = self._apply_rotary(q, k, cos, sin)
        
        # Handle past KV for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Try flash attention first
        if self.use_flash and seq_len > 1:
            try:
                # Flash Attention 2
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.config.attention_dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=True
                )
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                
            except:
                self.use_flash = False
                out = self._standard_attention(q, k, v, mask, seq_len)
        
        # Try xFormers
        elif self.use_xformers and seq_len > 1:
            try:
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                out = xops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=xops.LowerTriangularMask(),
                    p=self.config.attention_dropout if self.training else 0.0,
                    scale=self.scale
                )
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                
            except:
                self.use_xformers = False
                out = self._standard_attention(q, k, v, mask, seq_len)
        
        # Try PyTorch SDPA
        elif self.use_sdpa and seq_len > 1:
            try:
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.config.attention_dropout if self.training else 0.0,
                    is_causal=True,
                    scale=self.scale
                )
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                
            except:
                self.use_sdpa = False
                out = self._standard_attention(q, k, v, mask, seq_len)
        
        else:
            # Standard attention
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out = self._standard_attention(q, k, v, mask, seq_len)
        
        # Output projection
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        
        return out, (k, v)
    
    def _standard_attention(self, q, k, v, mask, seq_len):
        """Standard attention implementation"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device),
            diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(
            out.size(0), seq_len, -1
        )
        
        return out
    
    def _apply_rotary(self, q, k, cos, sin):
        """Apply rotary embeddings"""
        # Split into pairs
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        
        # Rotate
        q_rotated = torch.stack([
            q1 * cos - q2 * sin,
            q2 * cos + q1 * sin
        ], dim=-1).flatten(-2)
        
        k_rotated = torch.stack([
            k1 * cos - k2 * sin,
            k2 * cos + k1 * sin
        ], dim=-1).flatten(-2)
        
        return q_rotated, k_rotated


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Ultra-advanced transformer block"""
    
    def __init__(self, layer_id: int, config: UltraAdvancedConfig):
        super().__init__()
        self.layer_id = layer_id
        
        # Normalization
        if config.use_rms_norm:
            self.norm1 = RMSNorm(config.model_dim)
            self.norm2 = RMSNorm(config.model_dim)
        else:
            self.norm1 = nn.LayerNorm(config.model_dim)
            self.norm2 = nn.LayerNorm(config.model_dim)
        
        # Attention
        self.attention = MultiHeadAttention(config)
        
        # MLP
        if config.use_swiglu:
            self.mlp = SwiGLU(config.model_dim, config.ff_dim, config.dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.model_dim, config.ff_dim, bias=False),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_dim, config.model_dim, bias=False),
                nn.Dropout(config.dropout)
            )
        
        # Dropout
        self.dropout = nn.Dropout(config.residual_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention
        attn_out, past_kv = self.attention(self.norm1(x), mask, past_kv)
        x = x + self.dropout(attn_out)
        
        # Pre-norm MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        
        return x, past_kv


class UltraAdvancedTransformer(nn.Module):
    """Enterprise-grade transformer language model"""
    
    def __init__(self, config: UltraAdvancedConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.embedding_dropout = nn.Dropout(config.embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(i, config) for i in range(config.num_layers)
        ])
        
        # Final normalization
        if config.use_rms_norm:
            self.norm_f = RMSNorm(config.model_dim)
        else:
            self.norm_f = nn.LayerNorm(config.model_dim)
        
        # Output head (tied weights)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special initialization for residual layers
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * config.num_layers)
                )
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.use_checkpointing
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        self.num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)
        
        # Transformer blocks
        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            if self.gradient_checkpointing and self.training:
                x, new_past_kv = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask, past_kv,
                    use_reentrant=False
                )
            else:
                x, new_past_kv = block(x, attention_mask, past_kv)
            
            new_past_kvs.append(new_past_kv)
        
        # Final norm
        x = self.norm_f(x)
        
        # LM head
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return self.num_params
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return self.num_trainable


# =============================================================================
# TRAINING ENGINE
# =============================================================================

class UltraAdvancedTrainer:
    """Enterprise-grade training engine"""
    
    def __init__(
        self,
        model: UltraAdvancedTransformer,
        config: UltraAdvancedConfig,
        memory_manager: MemoryManager,
        logger: EnterpriseLogger,
        distributed_manager: DistributedManager
    ):
        self.model = model
        self.config = config
        self.memory_manager = memory_manager
        self.logger = logger
        self.distributed = distributed_manager
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Distributed
        if self.distributed.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.distributed.local_rank],
                output_device=self.distributed.local_rank
            )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # AMP
        self.scaler = GradScaler() if config.enable_amp and AMP_AVAILABLE else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self.best_loss = float('inf')
        
        # Tokenizer placeholder
        self.tokenizer = None
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                progress = (step - self.config.warmup_steps) / self.config.decay_steps
                return self.config.min_learning_rate + 0.5 * (1 - self.config.min_learning_rate) * (
                    1 + math.cos(math.pi * progress)
                )
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.scheduler_type == "linear":
            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                return max(
                    self.config.min_learning_rate,
                    1.0 - (step - self.config.warmup_steps) / self.config.decay_steps
                )
            return LambdaLR(self.optimizer, lr_lambda)
        
        else:
            return LambdaLR(self.optimizer, lambda step: 1.0)
    
    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting training...")
        
        # Create dataset
        dataset = StreamingDataset(
            file_pattern=f"{self.config.data_dir}/training_data_*.txt",
            tokenizer=self.tokenizer,
            config=self.config,
            shuffle=True,
            infinite=True
        )
        
        # Create data loader
        sampler = self.distributed.get_sampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.config.pin_memory,
            sampler=sampler
        )
        
        # Training loop
        self.model.train()
        iterator = iter(loader)
        
        while self.global_step < self.config.max_steps:
            try:
                # Get batch
                batch = next(iterator)
                
                # Train step
                loss = self.train_step(batch)
                
                # Logging
                if self.global_step % self.config.log_every_steps == 0:
                    self.log_metrics(loss)
                
                # Checkpointing
                if self.global_step % self.config.save_every_steps == 0 and self.distributed.is_main:
                    self.save_checkpoint()
                
                # Memory cleanup
                if self.global_step % self.config.empty_cache_freq == 0:
                    self.memory_manager.cleanup()
                
                self.global_step += 1
                
            except StopIteration:
                iterator = iter(loader)
            except Exception as e:
                self.logger.error(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Final checkpoint
        if self.distributed.is_main:
            self.save_checkpoint(final=True)
        
        self.logger.info("✅ Training completed!")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        with autocast(enabled=self.config.enable_amp and AMP_AVAILABLE):
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0
            )
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        # Track tokens
        self.tokens_seen += input_ids.numel()
        
        return loss.item()
    
    def log_metrics(self, loss: float):
        """Log training metrics"""
        lr = self.scheduler.get_last_lr()[0]
        
        metrics = {
            'loss': loss,
            'learning_rate': lr,
            'tokens_seen': self.tokens_seen,
            'step': self.global_step
        }
        
        self.logger.metrics_dict(metrics, step=self.global_step)
        
        if self.distributed.is_main:
            self.logger.info(
                f"Step {self.global_step:,} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tokens: {self.tokens_seen:,}"
            )
    
    def save_checkpoint(self, final: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if self.distributed.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'tokens_seen': self.tokens_seen,
            'best_loss': self.best_loss,
            'config': asdict(self.config)
        }
        
        # Save checkpoint
        if final:
            checkpoint_path = f"{self.config.checkpoint_dir}/checkpoint_final.pt"
        else:
            checkpoint_path = f"{self.config.checkpoint_dir}/checkpoint_step_{self.global_step:08d}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Keep only last N checkpoints
        if not final:
            checkpoints = sorted(glob.glob(f"{self.config.checkpoint_dir}/checkpoint_step_*.pt"))
            if len(checkpoints) > self.config.keep_last_n_checkpoints:
                for old in checkpoints[:-self.config.keep_last_n_checkpoints]:
                    os.remove(old)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict') and hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.global_step = checkpoint.get('global_step', 0)
            self.epoch = checkpoint.get('epoch', 0)
            self.tokens_seen = checkpoint.get('tokens_seen', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            self.logger.info(f"✅ Loaded checkpoint from step {self.global_step}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            return False


# =============================================================================
# BPE TOKENIZER TRAINING
# =============================================================================

def train_bpe_tokenizer(config: UltraAdvancedConfig, data_manager: AdvancedMultiFileDataManager):
    """Train BPE tokenizer on your data"""
    print("\n" + "="*60)
    print("🔤 TRAINING BPE TOKENIZER (200K VOCAB)")
    print("="*60)
    
    if not TOKENIZERS_AVAILABLE:
        print("❌ tokenizers library not available!")
        print("   Install with: pip install tokenizers")
        return None
    
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # Setup trainer
    trainer = trainers.BpeTrainer(
        vocab_size=200000,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )
    
    # Get all text files for training
    training_files = glob.glob(f"{config.data_dir}/*.txt")
    
    if not training_files:
        print("❌ No training files found in data directory!")
        print(f"   Looked in: {config.data_dir}")
        print("   Please add some text files to inbox and process them first.")
        return None
    
    print(f"📚 Training on {len(training_files)} files...")
    
    # Train the tokenizer
    tokenizer.train(files=training_files, trainer=trainer)
    
    # Save the tokenizer
    os.makedirs(config.bpe_checkpoint_dir, exist_ok=True)
    tokenizer_path = f"{config.bpe_checkpoint_dir}/tokenizer.json"
    tokenizer.save(tokenizer_path)
    
    print(f"✅ Tokenizer saved to: {tokenizer_path}")
    print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main execution pipeline"""
    
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║   ██╗   ██╗██╗  ████████╗██████╗  █████╗                              ║
    ║   ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗                             ║
    ║   ██║   ██║██║     ██║   ██████╔╝███████║                             ║
    ║   ██║   ██║██║     ██║   ██╔══██╗██╔══██║                             ║
    ║   ╚██████╔╝███████╗██║   ██║  ██║██║  ██║                             ║
    ║    ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝                             ║
    ║                                                                           ║
    ║              ULTRA-ADVANCED AI TRAINING SYSTEM                           ║
    ║                   ENTERPRISE EDITION v3.0                                ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Auto-detect capabilities
    print("\n🔍 Scanning system capabilities...")
    caps = SystemCapabilities()
    caps.print_summary()
    
    # Initialize configuration
    config = UltraAdvancedConfig()
    config.print_config()
    
    # Initialize memory manager
    memory_manager = MemoryManager(config)
    
    # Initialize distributed manager
    distributed = DistributedManager(config)
    
    # Initialize logger
    logger = EnterpriseLogger(config)
    
    # Create data directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.inbox_dir, exist_ok=True)
    
    # Initialize data manager
    data_manager = AdvancedMultiFileDataManager(config)
    
    # Process inbox files
    logger.info("📥 Scanning inbox for new files...")
    processed = data_manager.process_files(max_files=None)
    if processed > 0:
        logger.info(f"✅ Processed {processed} new files")
    
    # Check for data files
    data_files = glob.glob(f"{config.data_dir}/*.txt")
    if not data_files:
        logger.warning("No training data files found. Please add files to inbox.")
        logger.info(f"   Inbox directory: {config.inbox_dir}")
        logger.info("   Add text files there and run again.")
        return
    
    logger.info(f"Found {len(data_files)} training files")
    
    # Check for tokenizer or train one
    tokenizer_path = f"{config.bpe_checkpoint_dir}/tokenizer.json"
    if os.path.exists(tokenizer_path):
        if TOKENIZERS_AVAILABLE:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info(f"✅ Loaded tokenizer: {tokenizer.get_vocab_size():,} tokens")
        else:
            logger.error("❌ Tokenizers library not available!")
            return
    else:
        logger.warning("No tokenizer found. Training new BPE tokenizer...")
        tokenizer = train_bpe_tokenizer(config, data_manager)
        if tokenizer is None:
            logger.error("Failed to train tokenizer. Please add data first.")
            return
    
    # Initialize model
    logger.info("🏗️ Initializing model...")
    with memory_manager.track_allocation("model_creation"):
        model = UltraAdvancedTransformer(config)
    
    logger.info(f"   Parameters: {model.get_num_params() / 1e6:.1f}M")
    logger.info(f"   Trainable: {model.get_trainable_params() / 1e6:.1f}M")
    
    # Initialize trainer
    trainer = UltraAdvancedTrainer(
        model=model,
        config=config,
        memory_manager=memory_manager,
        logger=logger,
        distributed_manager=distributed
    )
    
    # Set tokenizer for trainer
    trainer.tokenizer = tokenizer
    
    # Check for existing checkpoint
    latest_checkpoint = f"{config.checkpoint_dir}/checkpoint_latest.pt"
    if os.path.exists(latest_checkpoint) and distributed.is_main:
        trainer.load_checkpoint(latest_checkpoint)
    
    # Train
    trainer.train()
    
    # Final cleanup
    memory_manager.shutdown()
    distributed.shutdown()
    logger.shutdown()
    
    print("\n" + "="*80)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
