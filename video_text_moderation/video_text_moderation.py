#!/usr/bin/env python3
"""
Video Text Moderation Script

This script performs end-to-end video text moderation by:
1. Extracting frames from videos
2. Using PaddleOCR to extract text from frames
3. Using Toxic-BERT to classify extracted text
4. Generating comprehensive reports

Author: AI Assistant
"""

import os
import json
import logging
import cv2
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# Import required libraries
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")


class VideoTextModerator:
    """Main class for video text moderation"""
    
    def __init__(self, 
                 toxic_bert_path: str = "models/toxic-bert",
                 paddle_model_dir: str = "models/paddleocr",
                 use_gpu: bool = True):
        """
        Initialize the Video Text Moderator
        
        Args:
            toxic_bert_path: Path to local Toxic-BERT model
            paddle_model_dir: Path to PaddleOCR model directory
            use_gpu: Whether to use GPU acceleration
        """
        self.toxic_bert_path = toxic_bert_path
        self.paddle_model_dir = paddle_model_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.ocr_model = None
        self.toxic_model = None
        self.toxic_tokenizer = None
        
        # Create necessary directories
        self.create_directories()
        
        # Load models
        self.load_models()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('video_moderation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['input_videos', 'frames', 'output', 'models/toxic-bert', 'models/paddleocr']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.logger.info("Created necessary directories")
    
    def load_models(self):
        """Load PaddleOCR and Toxic-BERT models"""
        try:
            # Load PaddleOCR
            self.logger.info("Loading PaddleOCR model...")
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.use_gpu,
                model_dir=self.paddle_model_dir if os.path.exists(self.paddle_model_dir) else None
            )
            self.logger.info("PaddleOCR model loaded successfully")
            
            # Load Toxic-BERT
            self.logger.info("Loading Toxic-BERT model...")
            device = 'cuda' if self.use_gpu else 'cpu'
            
            self.toxic_tokenizer = AutoTokenizer.from_pretrained(
                self.toxic_bert_path,
                local_files_only=True
            )
            self.toxic_model = AutoModelForSequenceClassification.from_pretrained(
                self.toxic_bert_path,
                local_files_only=True
            ).to(device)
            self.toxic_model.eval()
            
            self.logger.info(f"Toxic-BERT model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, output_dir: str, frame_interval: int = 30) -> List[str]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame
            
        Returns:
            List of extracted frame paths
        """
        frame_paths = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Processing video: {video_path}")
            self.logger.info(f"Total frames: {total_frames}, FPS: {fps}")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            self.logger.info(f"Extracted {extracted_count} frames from {os.path.basename(video_path)}")
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            
        return frame_paths
    
    def extract_text_from_frame(self, frame_path: str) -> List[Dict]:
        """
        Extract text from a single frame using PaddleOCR
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            List of text detection results
        """
        text_results = []
        
        try:
            result = self.ocr_model.ocr(frame_path, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        bbox = line[0]  # Bounding box coordinates
                        text_info = line[1]  # (text, confidence)
                        
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            if text and text.strip():  # Only process non-empty text
                                text_results.append({
                                    'text': text.strip(),
                                    'confidence': float(confidence),
                                    'bbox': bbox
                                })
                                
        except Exception as e:
            self.logger.error(f"Error extracting text from {frame_path}: {str(e)}")
            
        return text_results
    
    def classify_toxicity(self, text: str) -> Dict:
        """
        Classify text toxicity using Toxic-BERT
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with toxicity classification results
        """
        try:
            # Tokenize input
            inputs = self.toxic_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device if using GPU
            if self.use_gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.toxic_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
            # Convert to numpy for easier handling
            probs = probabilities.cpu().numpy()[0]
            
            # Assuming binary classification: [non-toxic, toxic]
            toxic_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            non_toxic_prob = float(probs[0]) if len(probs) > 1 else 1.0 - float(probs[0])
            
            is_toxic = toxic_prob > 0.5
            confidence = max(toxic_prob, non_toxic_prob)
            
            return {
                'is_toxic': is_toxic,
                'toxic_probability': toxic_prob,
                'non_toxic_probability': non_toxic_prob,
                'confidence': confidence,
                'label': 'TOXIC' if is_toxic else 'SAFE'
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying text toxicity: {str(e)}")
            return {
                'is_toxic': False,
                'toxic_probability': 0.0,
                'non_toxic_probability': 1.0,
                'confidence': 0.0,
                'label': 'ERROR',
                'error': str(e)
            }
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a single video for text moderation
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with complete analysis results
        """
        video_name = Path(video_path).stem
        video_frames_dir = os.path.join('frames', video_name)
        Path(video_frames_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting processing of video: {video_name}")
        
        # Extract frames
        frame_paths = self.extract_frames(video_path, video_frames_dir)
        
        if not frame_paths:
            self.logger.warning(f"No frames extracted from {video_name}")
            return {
                'video_name': video_name,
                'video_path': video_path,
                'status': 'error',
                'error': 'No frames extracted',
                'frames_analyzed': 0,
                'text_detections': 0,
                'toxic_detections': 0,
                'results': []
            }
        
        results = []
        total_text_detections = 0
        total_toxic_detections = 0
        
        for i, frame_path in enumerate(frame_paths):
            frame_name = os.path.basename(frame_path)
            self.logger.info(f"Processing frame {i+1}/{len(frame_paths)}: {frame_name}")
            
            # Extract text from frame
            text_results = self.extract_text_from_frame(frame_path)
            
            frame_result = {
                'frame_number': i,
                'frame_filename': frame_name,
                'frame_path': frame_path,
                'text_detections': []
            }
            
            for text_result in text_results:
                text = text_result['text']
                total_text_detections += 1
                
                # Classify toxicity
                toxicity_result = self.classify_toxicity(text)
                
                if toxicity_result['is_toxic']:
                    total_toxic_detections += 1
                
                detection = {
                    'text': text,
                    'ocr_confidence': text_result['confidence'],
                    'bbox': text_result['bbox'],
                    'toxicity_classification': toxicity_result
                }
                
                frame_result['text_detections'].append(detection)
            
            results.append(frame_result)
        
        # Calculate summary statistics
        summary = {
            'video_name': video_name,
            'video_path': video_path,
            'processing_timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'frames_analyzed': len(frame_paths),
            'total_text_detections': total_text_detections,
            'total_toxic_detections': total_toxic_detections,
            'toxicity_rate': total_toxic_detections / max(total_text_detections, 1),
            'overall_safety_status': 'UNSAFE' if total_toxic_detections > 0 else 'SAFE',
            'frame_results': results
        }
        
        self.logger.info(f"Completed processing {video_name}: "
                        f"{total_text_detections} text detections, "
                        f"{total_toxic_detections} toxic detections")
        
        return summary
    
    def save_report(self, video_result: Dict, output_path: str):
        """Save analysis report to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(video_result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
    
    def generate_summary_report(self, all_results: List[Dict], output_path: str):
        """Generate a summary report for all processed videos"""
        total_videos = len(all_results)
        total_frames = sum(r.get('frames_analyzed', 0) for r in all_results)
        total_text = sum(r.get('total_text_detections', 0) for r in all_results)
        total_toxic = sum(r.get('total_toxic_detections', 0) for r in all_results)
        unsafe_videos = sum(1 for r in all_results if r.get('overall_safety_status') == 'UNSAFE')
        
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'summary_statistics': {
                'total_videos_processed': total_videos,
                'total_frames_analyzed': total_frames,
                'total_text_detections': total_text,
                'total_toxic_detections': total_toxic,
                'unsafe_videos_count': unsafe_videos,
                'overall_toxicity_rate': total_toxic / max(total_text, 1),
                'video_safety_rate': (total_videos - unsafe_videos) / max(total_videos, 1)
            },
            'video_results': all_results
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Summary report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving summary report: {str(e)}")
    
    def process_all_videos(self, input_dir: str = "input_videos"):
        """Process all videos in the input directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        
        # Find all video files
        for ext in video_extensions:
            video_files.extend(Path(input_dir).glob(f"*{ext}"))
            video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not video_files:
            self.logger.warning(f"No video files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        all_results = []
        
        for video_file in video_files:
            try:
                # Process video
                result = self.process_video(str(video_file))
                all_results.append(result)
                
                # Save individual report
                output_filename = f"{result['video_name']}_moderation_report.json"
                output_path = os.path.join('output', output_filename)
                self.save_report(result, output_path)
                
            except Exception as e:
                self.logger.error(f"Error processing {video_file}: {str(e)}")
                error_result = {
                    'video_name': video_file.stem,
                    'video_path': str(video_file),
                    'status': 'error',
                    'error': str(e),
                    'processing_timestamp': datetime.now().isoformat()
                }
                all_results.append(error_result)
        
        # Generate summary report
        summary_path = os.path.join('output', 'moderation_summary_report.json')
        self.generate_summary_report(all_results, summary_path)
        
        self.logger.info("Video text moderation processing completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Video Text Moderation Script")
    parser.add_argument('--input-dir', default='input_videos',
                       help='Directory containing input videos')
    parser.add_argument('--toxic-bert-path', default='models/toxic-bert',
                       help='Path to Toxic-BERT model')
    parser.add_argument('--paddleocr-path', default='models/paddleocr',
                       help='Path to PaddleOCR model directory')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Extract every Nth frame (default: 30)')
    
    args = parser.parse_args()
    
    try:
        # Initialize moderator
        moderator = VideoTextModerator(
            toxic_bert_path=args.toxic_bert_path,
            paddle_model_dir=args.paddleocr_path,
            use_gpu=not args.no_gpu
        )
        
        # Process all videos
        moderator.process_all_videos(args.input_dir)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()