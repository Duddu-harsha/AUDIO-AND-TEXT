import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from models.asr_model import ASRModel
from models.safety_classifier import SafetyClassifier
from processors.video_processor import VideoProcessor
from processors.audio_processor import AudioProcessor
from utils.gpu_utils import GPUMonitor
from utils.output_formatter import OutputFormatter
from config import Config

class AudioSafetyPipeline:
    """Main processing pipeline for audio safety detection"""
    
    def __init__(self):
        self.config = Config()
        self.gpu_monitor = GPUMonitor()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.output_formatter = OutputFormatter()
        
        # Initialize models
        logger.info("Initializing processing pipeline...")
        self.asr_model = ASRModel()
        self.safety_classifier = SafetyClassifier()
        
        logger.success("Pipeline initialized successfully")
    
    def process_video(self, video_path: str) -> Dict:
        """Process a video file and return safety analysis"""
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting pipeline for: {video_path.name}")
        
        try:
            # Step 1: Extract video information
            video_info = self.video_processor.get_video_info(str(video_path))
            logger.debug(f"Video duration: {video_info['duration']}s")
            
            # Step 2: Extract audio
            audio_path = self.video_processor.extract_audio(str(video_path))
            logger.debug(f"Audio extracted to: {audio_path}")
            
            # Step 3: Preprocess audio
            processed_audio_path = self.audio_processor.preprocess_audio(audio_path)
            logger.debug("Audio preprocessing completed")
            
            # Step 4: Speech-to-text transcription
            logger.info("Starting speech recognition...")
            transcription_result = self.asr_model.transcribe_audio(processed_audio_path)
            logger.info(f"Transcribed {len(transcription_result['text'])} characters")
            
            # Step 5: Safety classification
            logger.info("Starting safety analysis...")
            if Config.OUTPUT_CONFIG["include_timestamps"] and transcription_result.get("segments"):
                safety_result = self.safety_classifier.classify_segments(
                    transcription_result["segments"],
                    transcription_result["text"]
                )
            else:
                safety_result = self.safety_classifier.classify_text(
                    transcription_result["text"]
                )
            
            # Step 6: Generate content summary
            content_summary = self._generate_content_summary(
                transcription_result, safety_result
            )
            
            # Step 7: Compile results
            total_processing_time = time.time() - start_time
            
            result = {
                "video_info": {
                    "filename": video_path.name,
                    "filepath": str(video_path),
                    "duration": video_info["duration"],
                    "size_mb": video_info["size_mb"],
                    "format": video_info["format"],
                    "processed_at": datetime.now(timezone.utc).isoformat()
                },
                "audio_analysis": {
                    "transcription": transcription_result["text"],
                    "language": transcription_result["language"],
                    "confidence": transcription_result.get("language_probability", 0.0),
                    "word_count": len(transcription_result["text"].split()),
                    "duration": transcription_result["duration"],
                    "segments_count": len(transcription_result.get("segments", []))
                },
                "safety_assessment": safety_result,
                "content_summary": content_summary,
                "technical_details": {
                    "processing_time": round(total_processing_time, 2),
                    "gpu_memory_used": f"{self.gpu_monitor.get_memory_usage():.1f}GB",
                    "models_used": {
                        "asr": self.asr_model.model_name,
                        "safety": [name for name in self.safety_classifier.models]
                    },
                    "pipeline_version": "1.0.0"
                }
            }
            
            # Add detailed segments if requested
            if (Config.OUTPUT_CONFIG["detailed_analysis"] and 
                Config.OUTPUT_CONFIG["include_timestamps"] and
                transcription_result.get("segments")):
                result["detailed_segments"] = transcription_result["segments"]
            
            # Cleanup temporary files
            self._cleanup_temp_files([audio_path, processed_audio_path])
            
            logger.success(f"Pipeline completed in {total_processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            # Cleanup on error
            self._cleanup_temp_files([audio_path, processed_audio_path] if 'audio_path' in locals() else [])
            raise
    
    def _generate_content_summary(self, transcription: Dict, safety: Dict) -> Dict:
        """Generate content summary from transcription and safety analysis"""
        text = transcription.get("text", "")
        
        # Extract key phrases (simple approach)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3 and word.isalpha():  # Filter short words and non-alpha
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top phrases
        key_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        key_phrases = [phrase[0] for phrase in key_phrases]
        
        # Simple sentiment analysis based on safety scores
        safety_score = safety.get("safety_score", 0.0)
        if safety_score > 0.7:
            sentiment = "very_negative"
        elif safety_score > 0.5:
            sentiment = "negative"
        elif safety_score > 0.3:
            sentiment = "mixed"
        else:
            sentiment = "neutral"
        
        # Estimate speaker count (simple approach based on sentence patterns)
        segments = transcription.get("segments", [])
        speaker_changes = 1  # At least one speaker
        if segments:
            # Look for conversation patterns
            for i in range(1, len(segments)):
                prev_text = segments[i-1]["text"].strip()
                curr_text = segments[i]["text"].strip()
                
                # Simple heuristic for speaker changes
                if (curr_text.startswith(("Yes", "No", "Well", "But", "However")) or
                    prev_text.endswith(("?", "."))) and len(curr_text.split()) < 20:
                    speaker_changes += 1
        
        # Determine topics based on detected issues and content
        topics = ["conversation"]  # default
        if safety.get("detected_issues"):
            issue_types = [issue["type"] for issue in safety["detected_issues"]]
            if any("hate" in issue_type for issue_type in issue_types):
                topics.append("controversial")
            if any("threat" in issue_type for issue_type in issue_types):
                topics.append("threatening")
            if any("toxic" in issue_type for issue_type in issue_types):
                topics.append("toxic_discussion")
        
        return {
            "topics": topics,
            "sentiment": sentiment,
            "key_phrases": key_phrases,
            "estimated_speaker_count": min(speaker_changes, 10),  # Cap at reasonable number
            "content_flags": [issue["type"] for issue in safety.get("detected_issues", [])],
            "language": transcription.get("language", "unknown")
        }
    
    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.debug(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {str(e)}")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline and loaded models"""
        return {
            "asr_model": self.asr_model.get_model_info(),
            "safety_models": [name for name in self.safety_classifier.models],
            "gpu_memory": f"{self.gpu_monitor.get_memory_usage():.1f}GB",
            "pipeline_version": "1.0.0"
        }