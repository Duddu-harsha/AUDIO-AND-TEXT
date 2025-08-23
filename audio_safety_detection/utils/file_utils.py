import os
import time
import shutil
from pathlib import Path
from typing import List, Optional
from loguru import logger

def get_video_files(directory: str) -> List[Path]:
    """Get all video files from a directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    video_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)

def ensure_output_path(output_path: str) -> Path:
    """Ensure output directory exists and return Path object"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)

def clean_temp_directory(temp_dir: str, max_age_hours: int = 24):
    """Clean old temporary files"""
    temp_dir = Path(temp_dir)
    current_time = time.time()
    
    for file_path in temp_dir.glob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_hours * 3600:  # Convert hours to seconds
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned old temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean {file_path}: {str(e)}")