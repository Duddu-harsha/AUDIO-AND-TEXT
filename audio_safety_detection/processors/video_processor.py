import ffmpeg
import os
from pathlib import Path
from typing import Dict
from loguru import logger
from config import Config

class VideoProcessor:
    """Handle video file processing and audio extraction"""
    
    def __init__(self):
        self.temp_dir = Config.TEMP_DIR
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            duration = float(probe['format']['duration'])
            size_bytes = int(probe['format']['size'])
            
            return {
                "duration": duration,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "format": probe['format']['format_name'],
                "video_codec": video_info.get('codec_name'),
                "audio_codec": audio_info.get('codec_name') if audio_info else None,
                "has_audio": audio_info is not None,
                "fps": eval(video_info.get('r_frame_rate', '0/1')),
                "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            raise
    
    def extract_audio(self, video_path: str, output_format: str = "wav") -> str:
        """Extract audio from video file"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Video format {video_path.suffix} may not be supported")
        
        # Generate output path
        audio_filename = f"{video_path.stem}_audio.{output_format}"
        audio_path = self.temp_dir / audio_filename
        
        try:
            logger.info(f"Extracting audio from {video_path.name}")
            
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    acodec='pcm_s16le',  # 16-bit PCM
                    ac=1,  # Mono
                    ar=16000,  # 16kHz sample rate (optimal for Whisper)
                    loglevel='error'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            if not audio_path.exists():
                raise RuntimeError("Audio extraction failed - output file not created")
            
            logger.success(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            raise