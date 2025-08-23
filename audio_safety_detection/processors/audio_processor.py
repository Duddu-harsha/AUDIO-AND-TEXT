import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple
from loguru import logger
from config import Config

class AudioProcessor:
    """Audio preprocessing and enhancement"""
    
    def __init__(self):
        self.temp_dir = Config.TEMP_DIR
        self.sample_rate = Config.AUDIO_CONFIG["sample_rate"]
        self.noise_reduction = Config.AUDIO_CONFIG["noise_reduction"]
        self.normalize_audio = Config.AUDIO_CONFIG["normalize_audio"]
    
    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio for optimal ASR performance"""
        audio_path = Path(audio_path)
        output_path = self.temp_dir / f"{audio_path.stem}_processed.wav"
        
        try:
            logger.debug(f"Preprocessing audio: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            
            # Apply preprocessing steps
            if self.noise_reduction:
                audio = self._reduce_noise(audio, sr)
            
            if self.normalize_audio:
                audio = self._normalize_audio(audio)
            
            # Trim silence
            audio = self._trim_silence(audio)
            
            # Save processed audio
            sf.write(str(output_path), audio, self.sample_rate)
            
            logger.debug(f"Audio preprocessing completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            # Return original file if preprocessing fails
            return str(audio_path)
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple noise reduction using spectral gating"""
        try:
            # Compute short-time Fourier transform
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor (bottom 10% of magnitude values)
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            # Create noise gate (reduce magnitude below threshold)
            gate_threshold = noise_floor * 2.0
            mask = magnitude > gate_threshold
            
            # Apply gate
            magnitude_filtered = magnitude * mask + magnitude * 0.1 * (1 - mask)
            
            # Reconstruct audio
            stft_filtered = magnitude_filtered * np.exp(1j * phase)
            audio_filtered = librosa.istft(stft_filtered, hop_length=512)
            
            return audio_filtered
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {str(e)}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        try:
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Leave some headroom
            
            return audio
            
        except Exception as e:
            logger.warning(f"Audio normalization failed: {str(e)}")
            return audio
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end"""
        try:
            # Find non-silent regions
            non_silent = np.abs(audio) > threshold
            
            if not np.any(non_silent):
                return audio  # All silent, return as-is
            
            # Find start and end of non-silent audio
            start_idx = np.argmax(non_silent)
            end_idx = len(audio) - np.argmax(non_silent[::-1]) - 1
            
            return audio[start_idx:end_idx + 1]
            
        except Exception as e:
            logger.warning(f"Silence trimming failed: {str(e)}")
            return audio