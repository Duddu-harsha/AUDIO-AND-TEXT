import argparse
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor, AutoModelForSpeechSeq2Seq
from detoxify import Detoxify
import whisper
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

from config import Config

def download_whisper_models():
    """Download and cache Whisper models"""
    models_to_download = ["base", "medium", "large-v3"]
    
    logger.info("Downloading Whisper models...")
    
    for model_size in models_to_download:
        try:
            logger.info(f"Downloading Whisper {model_size}...")
            model = whisper.load_model(model_size, download_root=str(Config.MODELS_DIR))
            logger.success(f"Whisper {model_size} downloaded successfully")
            del model  # Free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Failed to download Whisper {model_size}: {str(e)}")

def download_transformers_models():
    """Download Transformers models"""
    models_to_download = [
        "openai/whisper-large-v3",
        "openai/whisper-medium", 
        "unitary/toxic-bert",
        "martin-ha/toxic-comment-model",
        "KoalaAI/Text-Moderation"
    ]
    
    logger.info("Downloading Transformers models...")
    
    for model_name in models_to_download:
        try:
            logger.info(f"Downloading {model_name}...")
            
            if "whisper" in model_name:
                # Download Whisper models
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=str(Config.MODELS_DIR)
                )
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    cache_dir=str(Config.MODELS_DIR)
                )
            else:
                # Download classification models
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(Config.MODELS_DIR)
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=str(Config.MODELS_DIR)
                )
            
            logger.success(f"{model_name} downloaded successfully")
            del model, processor if 'processor' in locals() else None
            del tokenizer if 'tokenizer' in locals() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")

def download_detoxify_models():
    """Download Detoxify models"""
    logger.info("Downloading Detoxify models...")
    
    try:
        # This will download and cache the model
        detoxify = Detoxify('unitary/toxic-bert')
        logger.success("Detoxify model downloaded successfully")
        del detoxify
        
    except Exception as e:
        logger.error(f"Failed to download Detoxify model: {str(e)}")

def check_gpu_compatibility():
    """Check GPU compatibility and recommend optimal settings"""
    logger.info("Checking GPU compatibility...")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Models will run on CPU (slower performance)")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    if "A10G" in gpu_name or "A10" in gpu_name:
        logger.success("Optimal GPU detected for this system!")
    elif gpu_memory >= 16:
        logger.info("GPU has sufficient memory for all models")
    elif gpu_memory >= 8:
        logger.warning("GPU memory is adequate but may require model optimization")
    else:
        logger.warning("Limited GPU memory. Consider using smaller models")
    
    # Test GPU functionality
    try:
        test_tensor = torch.randn(100, 100).cuda()
        result = torch.mm(test_tensor, test_tensor)
        logger.success("GPU functionality test passed")
        del test_tensor, result
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"GPU functionality test failed: {str(e)}")

def setup_directories():
    """Create necessary directories"""
    directories = [
        Config.DATA_DIR,
        Config.INPUTS_DIR,
        Config.TEMP_DIR,
        Config.OUTPUTS_DIR,
        Config.MODELS_DIR,
        Config.CONFIGS_DIR,
        Config.BASE_DIR / "data" / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")
    
    logger.success("Directory structure created")

def create_sample_config():
    """Create sample configuration files"""
    # Model configs
    model_config = {
        "asr_models": {
            "primary": "openai/whisper-large-v3",
            "fallback": "openai/whisper-medium"
        },
        "safety_models": [
            "unitary/toxic-bert",
            "martin-ha/toxic-comment-model"
        ],
        "thresholds": {
            "toxicity": 0.7,
            "hate_speech": 0.6,
            "threat": 0.8
        }
    }
    
    config_path = Config.CONFIGS_DIR / "model_configs.yaml"
    
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        logger.success(f"Sample config created: {config_path}")
        
    except ImportError:
        logger.warning("PyYAML not available, skipping config file creation")
    except Exception as e:
        logger.error(f"Failed to create config file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Setup Audio Safety Detection System")
    parser.add_argument("--download-models", action="store_true", help="Download all required models")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU compatibility")
    parser.add_argument("--setup-dirs", action="store_true", help="Setup directory structure")
    parser.add_argument("--all", action="store_true", help="Run all setup tasks")
    
    args = parser.parse_args()
    
    if args.all:
        args.check_gpu = True
        args.setup_dirs = True
        args.download_models = True
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    logger.info("Starting Audio Safety Detection System setup...")
    
    if args.check_gpu:
        check_gpu_compatibility()
    
    if args.setup_dirs:
        setup_directories()
        create_sample_config()
    
    if args.download_models:
        logger.info("This may take several minutes depending on your internet connection...")
        download_whisper_models()
        download_transformers_models()
        download_detoxify_models()
        logger.success("Model download completed")
    
    logger.success("Setup completed successfully!")
    logger.info("You can now run: python main.py --video path/to/your/video.mp4")

if __name__ == "__main__":
    main()