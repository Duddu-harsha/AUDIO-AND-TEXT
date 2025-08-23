import torch
import psutil
from typing import Dict, Optional
from loguru import logger

class GPUMonitor:
    """Monitor and manage GPU resources"""
    
    def __init__(self):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.total_memory_gb = self._get_total_memory()
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        return torch.cuda.memory_allocated(self.device) / 1024**3
    
    def get_memory_info(self) -> Dict:
        """Get detailed GPU memory information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        total = self.total_memory_gb
        
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(self.device),
            "allocated_gb": round(allocated, 2),
            "cached_gb": round(cached, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
            "utilization_percent": round((allocated / total) * 100, 1)
        }
    
    def _get_total_memory(self) -> float:
        """Get total GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        return 0.0
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def check_memory_available(self, required_gb: float) -> bool:
        """Check if enough GPU memory is available"""
        if not torch.cuda.is_available():
            return False
        
        free_memory = self.total_memory_gb - self.get_memory_usage()
        return free_memory >= required_gb