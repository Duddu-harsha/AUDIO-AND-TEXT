import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from detoxify import Detoxify
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
from config import Config

class SafetyClassifier:
    """Multi-model safety and toxicity classifier"""
    
    def __init__(self, models: List[str] = None, device: str = None):
        self.device = device or Config.DEVICE
        self.models = models or [m["name"] for m in Config.select_optimal_models()["safety"]]
        self.classifiers = {}
        self.thresholds = Config.SAFETY_THRESHOLDS
        
        logger.info(f"Initializing safety classifiers: {self.models}")
        self._load_models()
    
    def _load_models(self):
        """Load all safety classification models"""
        for model_name in self.models:
            try:
                if "toxic-bert" in model_name or "unitary/toxic-bert" in model_name:
                    self.classifiers["toxicity"] = Detoxify("unitary/toxic-bert", device=self.device)
                    
                elif "martin-ha/toxic-comment-model" in model_name:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        torch_dtype=Config.TORCH_DTYPE
                    ).to(self.device)
                    
                    self.classifiers["hate_speech"] = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True
                    )
                    
                elif "KoalaAI/Text-Moderation" in model_name:
                    self.classifiers["content_moderation"] = pipeline(
                        "text-classification",
                        model=model_name,
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True
                    )
                
                logger.success(f"Loaded safety model: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue
    
    def classify_text(self, text: str, include_segments: bool = True) -> Dict:
        """Classify text for safety issues"""
        if not text or not text.strip():
            return self._empty_result()
        
        start_time = time.time()
        results = {
            "overall_safety": "SAFE",
            "risk_level": "SAFE", 
            "safety_score": 0.0,
            "detected_issues": [],
            "category_scores": {},
            "processing_time": 0.0
        }
        
        try:
            # Toxicity detection with Detoxify
            if "toxicity" in self.classifiers:
                toxicity_results = self.classifiers["toxicity"].predict(text)
                
                for category, score in toxicity_results.items():
                    results["category_scores"][category] = float(score)
                    
                    if score > self.thresholds.get(category, 0.7):
                        issue = {
                            "type": category,
                            "severity": self._get_severity(score),
                            "confidence": float(score),
                            "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                            "model": "unitary/toxic-bert"
                        }
                        results["detected_issues"].append(issue)
            
            # Hate speech detection
            if "hate_speech" in self.classifiers:
                hate_results = self.classifiers["hate_speech"](text)
                
                for result in hate_results:
                    label = result["label"].lower()
                    score = result["score"]
                    
                    results["category_scores"][f"hate_{label}"] = score
                    
                    if label == "hate" and score > self.thresholds.get("hate_speech", 0.6):
                        issue = {
                            "type": "hate_speech",
                            "severity": self._get_severity(score),
                            "confidence": score,
                            "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                            "model": "martin-ha/toxic-comment-model"
                        }
                        results["detected_issues"].append(issue)
            
            # Content moderation
            if "content_moderation" in self.classifiers:
                content_results = self.classifiers["content_moderation"](text)
                
                for result in content_results:
                    label = result["label"].lower()
                    score = result["score"]
                    
                    results["category_scores"][f"content_{label}"] = score
                    
                    if score > self.thresholds.get(label, 0.7):
                        issue = {
                            "type": f"content_{label}",
                            "severity": self._get_severity(score),
                            "confidence": score,
                            "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                            "model": "KoalaAI/Text-Moderation"
                        }
                        results["detected_issues"].append(issue)
            
            # Calculate overall safety
            results["safety_score"] = self._calculate_overall_score(results["category_scores"])
            results["risk_level"] = self._get_risk_level(results["safety_score"])
            results["overall_safety"] = "UNSAFE" if results["detected_issues"] else "SAFE"
            
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            
            logger.debug(f"Safety classification completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Safety classification failed: {str(e)}")
            raise
    
    def classify_segments(self, segments: List[Dict], text: str = None) -> Dict:
        """Classify individual text segments for timestamped issues"""
        all_issues = []
        category_scores = {}
        max_score = 0.0
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            if not segment_text:
                continue
            
            segment_result = self.classify_text(segment_text, include_segments=False)
            
            # Add timestamp information to issues
            for issue in segment_result["detected_issues"]:
                issue["timestamp"] = f"{segment['start']:.1f}-{segment['end']:.1f}s"
                issue["segment_start"] = segment["start"]
                issue["segment_end"] = segment["end"]
                all_issues.append(issue)
            
            # Aggregate scores
            for category, score in segment_result["category_scores"].items():
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
                max_score = max(max_score, score)
        
        # Calculate average scores
        avg_scores = {
            category: np.mean(scores) for category, scores in category_scores.items()
        }
        
        return {
            "overall_safety": "UNSAFE" if all_issues else "SAFE",
            "risk_level": self._get_risk_level(max_score),
            "safety_score": max_score,
            "detected_issues": all_issues,
            "category_scores": avg_scores,
            "segment_analysis": True
        }
    
    def _calculate_overall_score(self, category_scores: Dict) -> float:
        """Calculate overall safety score from category scores"""
        if not category_scores:
            return 0.0
        
        # Weight different categories
        weights = {
            "toxic": 1.0,
            "severe_toxic": 1.5,
            "threat": 1.3,
            "hate_speech": 1.2,
            "harassment": 1.1,
            "sexual": 0.9,
            "violence": 1.4
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = weights.get(category, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_severity(self, score: float) -> str:
        """Get severity level based on score"""
        if score >= 0.9:
            return "critical"
        elif score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on score"""
        for level, config in Config.RISK_LEVELS.items():
            if config["min_score"] <= score <= config["max_score"]:
                return level
        return "UNKNOWN"
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "overall_safety": "SAFE",
            "risk_level": "SAFE",
            "safety_score": 0.0,
            "detected_issues": [],
            "category_scores": {},
            "processing_time": 0.0
        }