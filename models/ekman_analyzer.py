# models/ekman_analyzer.py
"""
Ekman-Emotionen Analyse Koordinator
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from analyzers.base_analyzer import AnalysisResult
from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS
from utils.api_manager import APIManager

class EkmanAnalyzer:
    """Koordiniert Ekman-Emotionen Analyse mit Synonym-Clustering"""
    
    def __init__(self):
        self.api_manager = APIManager()
        self.analyzers = {}
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialisiert verfügbare Analyzer für Ekman-Emotionen"""
        try:
            # OpenAI
            config = self.api_manager.get_api_config("openai_reasoning")
            if config.primary_key:
                self.analyzers["apt-5-nano"] = OpenAIAnalyzer(config)
        except Exception:
            pass
        
        try:
            # DeepSeek
            config = self.api_manager.get_api_config("deepseek")
            if config.primary_key:
                self.analyzers["deepseek-chat"] = DeepSeekAnalyzer(config)
        except Exception:
            pass
        
        try:
            # HuggingFace Modelle (BART und RoBERTa, nicht SiEBERT)
            config = self.api_manager.get_api_config("huggingface")
            if config.primary_key:
                for model_name in ["facebook/bart-large", "FacebookAI/roberta-base"]:
                    self.analyzers[model_name] = HuggingFaceAnalyzer(model_name, config)
        except Exception:
            pass
    
    def analyze_single(self, text: str, models: Optional[List[str]] = None, **kwargs) -> Dict[str, AnalysisResult]:
        """Analysiert einen Text mit ausgewählten Modellen"""
        if models is None:
            models = list(self.analyzers.keys())
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(5, len(models))) as executor:
            futures = {}
            
            for model_name in models:
                if model_name in self.analyzers:
                    analyzer = self.analyzers[model_name]
                    if analyzer.is_available():
                        future = executor.submit(analyzer.analyze_single, text, "ekman", **kwargs)
                        futures[future] = model_name
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result(timeout=45)  # Längere Timeout für Ekman
                    # Wende Synonym-Clustering an
                    clustered_result = self._apply_synonym_clustering(result)
                    results[model_name] = clustered_result
                except Exception as e:
                    results[model_name] = AnalysisResult(
                        text=text,
                        model=model_name,
                        analysis_type="ekman",
                        scores={emotion: 0.0 for emotion in EKMAN_EMOTIONS.keys()},
                        processing_time=0,
                        error=str(e)
                    )
        
        return results
    
    def analyze_batch(self, texts: List[str], models: Optional[List[str]] = None, **kwargs) -> List[Dict[str, AnalysisResult]]:
        """Analysiert eine Liste von Texten"""
        if models is None:
            models = list(self.analyzers.keys())
        
        results = []
        for text in texts:
            text_results = self.analyze_single(text, models, **kwargs)
            results.append(text_results)
        
        return results
    
    def _apply_synonym_clustering(self, result: AnalysisResult) -> AnalysisResult:
        """Wendet Synonym-Clustering auf die Ergebnisse an"""
        if result.error:
            return result
        
        # Erstelle erweiterte Scores basierend auf Synonymen
        enhanced_scores = {}
        
        for emotion_key in EKMAN_EMOTIONS.keys():
            base_score = result.scores.get(emotion_key, 0.0)
            synonym_boost = 0.0
            
            # Prüfe Synonyme in den Original-Scores
            emotion_data = EKMAN_EMOTIONS[emotion_key]
            for synonym in emotion_data["synonyms"]:
                if synonym in result.scores:
                    synonym_boost += result.scores[synonym] * 0.3  # 30% Gewichtung für Synonyme
            
            # Kombiniere Base-Score mit Synonym-Boost
            enhanced_scores[emotion_key] = min(1.0, base_score + synonym_boost)
        
        # Normalisiere Scores
        total_score = sum(enhanced_scores.values())
        if total_score > 0:
            enhanced_scores = {k: v / total_score for k, v in enhanced_scores.items()}
        
        # Erstelle neues Ergebnis mit enhanced scores
        return AnalysisResult(
            text=result.text,
            model=result.model,
            analysis_type=result.analysis_type,
            scores=enhanced_scores,
            processing_time=result.processing_time,
            metadata={
                "original_scores": result.scores,
                "synonym_clustering_applied": True
            }
        )
    
    def get_available_models(self) -> List[str]:
        """Gibt verfügbare Modelle für Ekman-Analyse zurück"""
        return [name for name, analyzer in self.analyzers.items() if analyzer.is_available()]
    
    def get_aggregated_scores(self, results: Dict[str, AnalysisResult]) -> Dict[str, float]:
        """Aggregiert Scores über alle Modelle"""
        emotion_scores = {emotion: [] for emotion in EKMAN_EMOTIONS.keys()}
        
        for model_name, result in results.items():
            if not result.error:
                for emotion, score in result.scores.items():
                    if emotion in emotion_scores:
                        emotion_scores[emotion].append(score)
        
        # Berechne Durchschnittswerte
        aggregated = {}
        for emotion, scores in emotion_scores.items():
            if scores:
                aggregated[emotion] = np.mean(scores)
            else:
                aggregated[emotion] = 0.0
        
        return aggregated
