import argparse
import json
import time
from pathlib import Path
from typing import List, Optional
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "data/logs/audio_safety_{time}.log",
    rotation="100 MB",
    level="DEBUG"
)

from processors.pipeline import AudioSafetyPipeline
from utils.file_utils import get_video_files, ensure_output_path
from config import Config

def process_single_video(video_path: str, output_path: Optional[str] = None) -> dict:
    """Process a single video file"""
    logger.info(f"Processing video: {video_path}")
    
    pipeline = AudioSafetyPipeline()
    
    start_time = time.time()
    try:
        result = pipeline.process_video(video_path)
        processing_time = time.time() - start_time
        
        result["technical_details"]["processing_time"] = round(processing_time, 2)
        
        # Save output
        if not output_path:
            video_name = Path(video_path).stem
            output_path = Config.OUTPUTS_DIR / f"{video_name}_analysis.json"
        
        ensure_output_path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Analysis completed in {processing_time:.2f}s")
        logger.info(f"Results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")
        raise

def process_batch(input_dir: str, output_dir: str) -> List[dict]:
    """Process multiple video files"""
    video_files = get_video_files(input_dir)
    
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    results = []
    pipeline = AudioSafetyPipeline()
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"Processing {i}/{len(video_files)}: {video_path}")
        
        try:
            result = pipeline.process_video(str(video_path))
            
            # Save individual result
            video_name = video_path.stem
            output_path = Path(output_dir) / f"{video_name}_analysis.json"
            ensure_output_path(output_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            results.append(result)
            logger.success(f"Completed {video_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {str(e)}")
            continue
    
    # Save batch summary
    summary_path = Path(output_dir) / "batch_summary.json"
    batch_summary = {
        "total_videos": len(video_files),
        "processed_successfully": len(results),
        "failed": len(video_files) - len(results),
        "overall_safety_distribution": analyze_batch_safety(results),
        "processing_stats": calculate_batch_stats(results)
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Batch processing complete. Summary saved to {summary_path}")
    return results

def analyze_batch_safety(results: List[dict]) -> dict:
    """Analyze safety distribution across batch"""
    safety_counts = {"SAFE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    
    for result in results:
        risk_level = result.get("safety_assessment", {}).get("risk_level", "UNKNOWN")
        if risk_level in safety_counts:
            safety_counts[risk_level] += 1
    
    total = len(results)
    return {
        "counts": safety_counts,
        "percentages": {k: round(v/total*100, 1) if total > 0 else 0 
                      for k, v in safety_counts.items()}
    }

def calculate_batch_stats(results: List[dict]) -> dict:
    """Calculate processing statistics"""
    if not results:
        return {}
    
    processing_times = [r.get("technical_details", {}).get("processing_time", 0) 
                       for r in results]
    durations = [r.get("video_info", {}).get("duration", 0) for r in results]
    
    return {
        "avg_processing_time": round(sum(processing_times) / len(processing_times), 2),
        "total_processing_time": round(sum(processing_times), 2),
        "avg_video_duration": round(sum(durations) / len(durations), 2),
        "total_video_duration": round(sum(durations), 2),
        "processing_speed_ratio": round(sum(durations) / sum(processing_times), 2) if sum(processing_times) > 0 else 0
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Audio Safety Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video path/to/video.mp4
  python main.py --video video.mp4 --output results.json
  python main.py --input-dir videos/ --output-dir results/
  python main.py --batch videos/ results/
        """
    )
    
    # Single video processing
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    # Batch processing
    parser.add_argument("--input-dir", type=str, help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--batch", nargs=2, metavar=("INPUT_DIR", "OUTPUT_DIR"),
                       help="Batch process videos (input_dir output_dir)")
    
    # Options
    parser.add_argument("--detailed", action="store_true", help="Include detailed analysis")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                       help="Minimum confidence threshold")
    parser.add_argument("--gpu-info", action="store_true", help="Show GPU information and exit")
    
    args = parser.parse_args()
    
    # Show GPU info
    if args.gpu_info:
        gpu_memory = Config.get_gpu_memory()
        optimal_models = Config.select_optimal_models()
        
        print(f"GPU Available: {torch.cuda.is_available()}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"Selected ASR Model: {optimal_models['asr']['name']}")
        print(f"Selected Safety Models: {[m['name'] for m in optimal_models['safety']]}")
        print(f"Estimated Memory Usage: {optimal_models['estimated_memory_usage']:.1f} GB")
        return
    
    # Process single video
    if args.video:
        if not Path(args.video).exists():
            logger.error(f"Video file not found: {args.video}")
            return
        
        result = process_single_video(args.video, args.output)
        
        # Print summary
        safety = result["safety_assessment"]
        print(f"\n{'='*50}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"File: {result['video_info']['filename']}")
        print(f"Duration: {result['video_info']['duration']}s")
        print(f"Safety: {safety['overall_safety']} ({safety['risk_level']})")
        print(f"Confidence: {safety['safety_score']:.2f}")
        if safety['detected_issues']:
            print(f"Issues Found: {len(safety['detected_issues'])}")
        print(f"{'='*50}")
        
        return
    
    # Process batch
    if args.batch:
        input_dir, output_dir = args.batch
        results = process_batch(input_dir, output_dir)
        print(f"Processed {len(results)} videos successfully")
        return
    
    if args.input_dir and args.output_dir:
        results = process_batch(args.input_dir, args.output_dir)
        print(f"Processed {len(results)} videos successfully")
        return
    
    # No arguments provided
    parser.print_help()

if __name__ == "__main__":
    main()