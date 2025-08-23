import json
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path
from loguru import logger

class OutputFormatter:
    """Format and save analysis results"""
    
    def format_result(self, result: Dict) -> Dict:
        """Format result with consistent structure and readable output"""
        
        # Create formatted output
        formatted = {
            "analysis_summary": {
                "filename": result["video_info"]["filename"],
                "safety_verdict": result["safety_assessment"]["overall_safety"],
                "risk_level": result["safety_assessment"]["risk_level"],
                "confidence": round(result["safety_assessment"]["safety_score"], 3),
                "processing_time": f"{result['technical_details']['processing_time']}s",
                "timestamp": result["video_info"]["processed_at"]
            },
            
            "video_details": {
                "duration": f"{result['video_info']['duration']:.1f} seconds",
                "size": f"{result['video_info']['size_mb']} MB",
                "format": result["video_info"]["format"]
            },
            
            "transcription": {
                "full_text": result["audio_analysis"]["transcription"],
                "language": result["audio_analysis"]["language"],
                "word_count": result["audio_analysis"]["word_count"],
                "transcription_confidence": round(result["audio_analysis"]["confidence"], 3)
            },
            
            "safety_analysis": self._format_safety_analysis(result["safety_assessment"]),
            
            "content_insights": {
                "topics": result["content_summary"]["topics"],
                "sentiment": result["content_summary"]["sentiment"],
                "key_phrases": result["content_summary"]["key_phrases"][:5],  # Top 5
                "estimated_speakers": result["content_summary"]["estimated_speaker_count"]
            },
            
            "technical_info": {
                "models_used": result["technical_details"]["models_used"],
                "gpu_memory_used": result["technical_details"]["gpu_memory_used"],
                "processing_speed": self._calculate_processing_speed(result)
            }
        }
        
        # Add detailed segments if available
        if result.get("detailed_segments"):
            formatted["detailed_transcript"] = self._format_segments(result["detailed_segments"])
        
        return formatted
    
    def _format_safety_analysis(self, safety_data: Dict) -> Dict:
        """Format safety analysis with human-readable descriptions"""
        
        # Risk level descriptions
        risk_descriptions = {
            "SAFE": "Content appears safe with no significant safety concerns detected.",
            "LOW": "Minor safety concerns detected. Content may contain mild inappropriate language.",
            "MEDIUM": "Moderate safety concerns. Content contains potentially problematic material.",
            "HIGH": "Significant safety concerns. Content contains inappropriate or harmful material.",
            "CRITICAL": "Severe safety concerns. Content contains highly problematic or dangerous material."
        }
        
        formatted_issues = []
        for issue in safety_data.get("detected_issues", []):
            formatted_issue = {
                "issue_type": issue["type"].replace("_", " ").title(),
                "severity": issue["severity"].upper(),
                "confidence": f"{issue['confidence']:.1%}",
                "description": self._get_issue_description(issue["type"]),
                "content_preview": issue["text_snippet"][:100] + "..." if len(issue["text_snippet"]) > 100 else issue["text_snippet"]
            }
            
            # Add timestamp if available
            if "timestamp" in issue:
                formatted_issue["timestamp"] = issue["timestamp"]
            
            formatted_issues.append(formatted_issue)
        
        return {
            "overall_verdict": safety_data["overall_safety"],
            "risk_level": safety_data["risk_level"],
            "risk_description": risk_descriptions.get(safety_data["risk_level"], "Unknown risk level"),
            "safety_score": f"{safety_data['safety_score']:.1%}",
            "issues_detected": len(safety_data.get("detected_issues", [])),
            "detailed_issues": formatted_issues,
            "category_breakdown": self._format_category_scores(safety_data.get("category_scores", {}))
        }
    
    def _format_segments(self, segments: list) -> list:
        """Format transcript segments with timestamps"""
        formatted_segments = []
        
        for segment in segments:
            formatted_segment = {
                "start_time": f"{segment['start']:.1f}s",
                "end_time": f"{segment['end']:.1f}s",
                "duration": f"{segment['end'] - segment['start']:.1f}s",
                "text": segment["text"].strip(),
                "confidence": f"{segment.get('confidence', 0):.1%}"
            }
            formatted_segments.append(formatted_segment)
        
        return formatted_segments
    
    def _format_category_scores(self, category_scores: Dict) -> Dict:
        """Format category scores with percentages"""
        formatted_scores = {}
        
        for category, score in category_scores.items():
            readable_name = category.replace("_", " ").title()
            formatted_scores[readable_name] = f"{score:.1%}"
        
        return formatted_scores
    
    def _get_issue_description(self, issue_type: str) -> str:
        """Get human-readable description for issue types"""
        descriptions = {
            "toxic": "Content contains toxic or offensive language",
            "severe_toxic": "Content contains severely toxic or extremely offensive material",
            "obscene": "Content contains obscene or vulgar language",
            "threat": "Content contains threatening language or implications of harm",
            "insult": "Content contains insulting or derogatory language",
            "identity_hate": "Content contains hate speech targeting specific identity groups",
            "hate_speech": "Content contains hate speech or discriminatory language",
            "harassment": "Content contains harassing or bullying language",
            "sexual": "Content contains sexual or inappropriate material",
            "violence": "Content contains violent themes or descriptions",
            "profanity": "Content contains profane or inappropriate language"
        }
        
        return descriptions.get(issue_type.lower(), f"Content flagged for {issue_type}")
    
    def _calculate_processing_speed(self, result: Dict) -> str:
        """Calculate and format processing speed"""
        duration = result["video_info"]["duration"]
        processing_time = result["technical_details"]["processing_time"]
        
        if processing_time > 0:
            speed_ratio = duration / processing_time
            return f"{speed_ratio:.1f}x realtime"
        
        return "Unknown"
    
    def save_formatted_result(self, result: Dict, output_path: str):
        """Save formatted result to JSON file"""
        formatted_result = self.format_result(result)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_result, f, indent=2, ensure_ascii=False)
            
            logger.success(f"Formatted results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save formatted results: {str(e)}")
            raise