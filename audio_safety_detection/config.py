import torch
import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml

class Config:
    """Configuration class for audio safety detection system"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    INPUTS_DIR = DATA_DIR / "inputs"
    TEMP_DIR = DATA_DIR / "temp"
    OUTPUTS_DIR = DATA_DIR / "outputs"
    MODELS_DIR = DATA_DIR / "models"
    CONFIGS_DIR = BASE_DIR / "configs"
    
    # Create directories
    for dir_path in [DATA_DIR, INPUTS_DIR, TEMP_DIR, OUTPUTS_DIR, MODELS_DIR, CONFIGS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # GPU and Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_MEMORY_GB = 22  # A10G GPU memory minus headroom
    TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    
    # Model Selection (Best for A10G)
    ASR_MODELS = {
        "primary": {
            "name": "openai/whisper-large-v3",
            "vram_gb": 3.0,
            "batch_size": 8,
            "accuracy": "highest"
        },
        "secondary": {
            "name": "openai/whisper-medium",
            "vram_gb": 1.5,
            "batch_size": 16,
            "accuracy": "high"
        },
        "fallback": {
            "name": "openai/whisper-base",
            "vram_gb": 0.8,
            "batch_size": 32,
            "accuracy": "medium"
        }
    }
    
    SAFETY_MODELS = {
        "toxicity": {
            "name": "unitary/toxic-bert",
            "vram_gb": 0.5,
            "batch_size": 32,
            "categories": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        },
        "hate_speech": {
            "name": "martin-ha/toxic-comment-model", 
            "vram_gb": 0.4,
            "batch_size": 64,
            "categories": ["hate_speech", "harassment"]
        },
        "content_safety": {
            "name": "KoalaAI/Text-Moderation",
            "vram_gb": 0.3,
            "batch_size": 64,
            "categories": ["sexual", "violence", "profanity", "harassment"]
        }
    }
    
    # Processing Configuration
    AUDIO_CONFIG = {
        "sample_rate": 16000,
        "chunk_duration": 30,  # seconds
        "overlap": 2,  # seconds
        "noise_reduction": True,
        "normalize_audio": True
    }
    
    # Safety Thresholds
    SAFETY_THRESHOLDS = {
        "toxicity": 0.7,
        "severe_toxicity": 0.8,
        "hate_speech": 0.6,
        "harassment": 0.65,
        "threat": 0.8,
        "sexual": 0.7,
        "violence": 0.75,
        "profanity": 0.5
    }
    
    # Risk Levels
    RISK_LEVELS = {
        "SAFE": {"min_score": 0.0, "max_score": 0.3, "color": "green"},
        "LOW": {"min_score": 0.3, "max_score": 0.5, "color": "yellow"},
        "MEDIUM": {"min_score": 0.5, "max_score": 0.7, "color": "orange"},
        "HIGH": {"min_score": 0.7, "max_score": 0.9, "color": "red"},
        "CRITICAL": {"min_score": 0.9, "max_score": 1.0, "color": "darkred"}
    }
    
    # Output Configuration
    OUTPUT_CONFIG = {
        "include_timestamps": True,
        "include_confidence_scores": True,
        "include_technical_details": True,
        "detailed_analysis": True,
        "save_audio_segments": False,
        "language_detection": True
    }

    @classmethod
    def select_optimal_models(cls) -> Dict:
        """Select optimal models based on available GPU memory"""
        available_memory = cls.get_gpu_memory()
        
        # Select ASR model
        asr_model = cls.ASR_MODELS["fallback"]  # default
        for model_key in ["primary", "secondary", "fallback"]:
            model = cls.ASR_MODELS[model_key]
            if model["vram_gb"] <= available_memory * 0.7:  # 70% of available memory
                asr_model = model
                break
        
        # Select safety models (can run multiple)
        safety_models = []
        remaining_memory = available_memory - asr_model["vram_gb"]
        
        for model_key, model in cls.SAFETY_MODELS.items():
            if model["vram_gb"] <= remaining_memory:
                safety_models.append(model)
                remaining_memory -= model["vram_gb"]
        
        return {
            "asr": asr_model,
            "safety": safety_models,
            "estimated_memory_usage": available_memory - remaining_memory
        }
    
    @staticmethod
    def get_gpu_memory() -> float:
        """Get available GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0