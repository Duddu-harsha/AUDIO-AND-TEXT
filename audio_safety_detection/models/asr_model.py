import torch
import whisper
from faster_whisper import WhisperModel
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
from config import Config

class ASRModel:
    """Automatic Speech Recognition using Whisper models"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.device = device or Config.DEVICE
        self.model_name = model_name or Config.select_optimal_models()["asr"]["name"]
        self.model = None
        self.processor = None
        self.model_type = self._determine_model_type()
        
        logger.info(f"Initializing ASR model: {self.model_name}")
        self._load_model()
    
    def _determine_model_type(self) -> str:
        """Determine which Whisper implementation to use"""
        if "faster-whisper" in self.model_name.lower():
            return "faster_whisper"
        elif "openai/whisper" in self.model_name or "whisper" in self.model_name:
            return "transformers_whisper"
        else:
            return "openai_whisper"
    
    def _load_model(self):
        """Load the appropriate Whisper model"""
        try:
            if self.model_type == "faster_whisper":
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "float32"
                )
                
            elif self.model_type == "transformers_whisper":
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=Config.TORCH_DTYPE,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(self.device)
                
            else:  # openai_whisper
                model_size = self.model_name.split("-")[-1] if "-" in self.model_name else self.model_name
                self.model = whisper.load_model(model_size, device=self.device)
            
            logger.success(f"ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """Transcribe audio file to text"""
        start_time = time.time()
        
        try:
            if self.model_type == "faster_whisper":
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    task="transcribe",
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        max_speech_duration_s=30,
                        min_silence_duration_ms=100
                    )
                )
                
                transcription_segments = []
                full_text = ""
                
                for segment in segments:
                    segment_data = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "confidence": getattr(segment, 'avg_logprob', 0.0)
                    }
                    transcription_segments.append(segment_data)
                    full_text += segment.text + " "
                
                result = {
                    "text": full_text.strip(),
                    "segments": transcription_segments,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "all_language_probs": info.all_language_probs
                }
                
            elif self.model_type == "transformers_whisper":
                # Load audio
                import librosa
                audio, _ = librosa.load(audio_path, sr=16000)
                
                # Process
                input_features = self.processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        max_new_tokens=448,
                        do_sample=False,
                        suppress_tokens=[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 42, 50257]
                    )
                
                # Decode
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                result = {
                    "text": transcription.strip(),
                    "segments": [{"start": 0, "end": len(audio)/16000, "text": transcription.strip(), "confidence": 0.9}],
                    "language": language or "auto",
                    "language_probability": 0.9,
                    "duration": len(audio) / 16000
                }
                
            else:  # openai_whisper
                result_whisper = self.model.transcribe(
                    audio_path,
                    language=language,
                    task="transcribe",
                    verbose=False
                )
                
                segments = []
                for segment in result_whisper.get("segments", []):
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"], 
                        "text": segment["text"].strip(),
                        "confidence": getattr(segment, 'avg_logprob', 0.0)
                    })
                
                result = {
                    "text": result_whisper["text"].strip(),
                    "segments": segments,
                    "language": result_whisper.get("language", "unknown"),
                    "language_probability": 0.9,
                    "duration": segments[-1]["end"] if segments else 0
                }
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            logger.debug(f"Detected language: {result['language']} (confidence: {result['language_probability']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0