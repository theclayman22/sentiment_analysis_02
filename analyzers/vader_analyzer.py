# analyzers/vader_analyzer.py
"""
VADER Sentiment Analyzer
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any
import time

from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult

class VADERAnalyzer(BaseAnalyzer):
    """VADER Sentiment Analyzer"""
    
    def __init__(self, api_config=None):
        super().__init__("vader", api_config)
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        """Analysiert einen einzelnen Text"""
        try:
            start_time = time.time()
            
            if analysis_type == "valence":
                scores = self._analyze_valence(text, **kwargs)
            elif analysis_type == "ekman":
                # VADER unterstützt nur Valence
                scores = {emotion: 0.1 for emotion in ["joy", "surprise", "fear", "anger", "disgust", "sadness", "contempt"]}
            elif analysis_type == "emotion_arc":
                scores = self._analyze_happiness(text, **kwargs)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                text=text,
                model=self.model_name,
                analysis_type=analysis_type,
                scores=scores,
                processing_time=processing_time
            )
            
        except Exception as e:
            return AnalysisResult(
                text=text,
                model=self.model_name,
                analysis_type=analysis_type,
                scores={},
                processing_time=0,
                error=str(e)
            )
    
    def analyze_batch(self, texts: List[str], analysis_type: str, **kwargs) -> List[AnalysisResult]:
        """Analysiert eine Liste von Texten schnell"""
        results = []
        for text in texts:
            result = self.analyze_single(text, analysis_type, **kwargs)
            results.append(result)
        return results
    
    def is_available(self) -> bool:
        """VADER ist immer verfügbar"""
        return True
    
    def _analyze_valence(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Valence mit VADER"""
        scores = self.analyzer.polarity_scores(text)
        
        # VADER gibt compound, pos, neu, neg zurück
        # Normalisiere auf positive, negative, neutral
        positive = scores['pos']
        negative = scores['neg'] 
        neutral = scores['neu']
        
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        }
    
    def _analyze_happiness(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Happiness für Emotion Arc"""
        scores = self.analyzer.polarity_scores(text)
        
        # Verwende compound score und konvertiere zu 0-1
        compound = scores['compound']  # -1 bis 1
        happiness = (compound + 1) / 2  # 0 bis 1
        
        return {"happiness": happiness}
