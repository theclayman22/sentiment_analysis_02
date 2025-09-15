# models/valence_analyzer.py
"""
Valence-Analyse Koordinator
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from analyzers.base_analyzer import AnalysisResult
from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.vader_analyzer import VADERAnalyzer
from config.settings import Settings
from utils.api_manager import APIManager

class ValenceAnalyzer:
    """Koordiniert Valence-Analyse über alle Modelle"""
    
    def __init__(self):
        self.api_manager = APIManager()
        self.analyzers = {}
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialisiert alle verfügbaren Analyzer"""
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
            # HuggingFace Modelle
            config = self.api_manager.get_api_config("huggingface")
            if config.primary_key:
                for model_name in ["facebook/bart-large", "FacebookAI/roberta-base", "siebert/sentiment-roberta-large-english"]:
                    self.analyzers[model_name] = HuggingFaceAnalyzer(model_name, config)
        except Exception:
            pass
        
        # VADER (immer verfügbar)
        self.analyzers["vader"] = VADERAnalyzer()
    
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
                        future = executor.submit(analyzer.analyze_single, text, "valence", **kwargs)
                        futures[future] = model_name
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result(timeout=30)
                    results[model_name] = result
                except Exception as e:
                    results[model_name] = AnalysisResult(
                        text=text,
                        model=model_name,
                        analysis_type="valence",
                        scores={},
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
    
    def get_available_models(self) -> List[str]:
        """Gibt verfügbare Modelle zurück"""
        return [name for name, analyzer in self.analyzers.items() if analyzer.is_available()]
