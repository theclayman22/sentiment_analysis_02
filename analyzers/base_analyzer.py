# analyzers/base_analyzer.py
"""
Abstrakte Basisklasse für alle Sentiment-Analyzer
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import time
import logging
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """Standardisiertes Ergebnis-Format für alle Analyzer"""
    text: str
    model: str
    analysis_type: str
    scores: Dict[str, float]
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BaseAnalyzer(ABC):
    """Abstrakte Basisklasse für alle Sentiment-Analyzer"""
    
    def __init__(self, model_name: str, api_config: Any):
        self.model_name = model_name
        self.api_config = api_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        """Analysiert einen einzelnen Text"""
        pass
    
    @abstractmethod
    def analyze_batch(self, texts: List[str], analysis_type: str, **kwargs) -> List[AnalysisResult]:
        """Analysiert eine Liste von Texten"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Prüft, ob der Analyzer verfügbar ist"""
        pass
    
    def _measure_time(self, func, *args, **kwargs):
        """Misst die Ausführungszeit einer Funktion"""
        start_time = time.time()
        result = func(*args, **kwargs)
        processing_time = time.time() - start_time
        return result, processing_time
