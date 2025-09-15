# analyzers/huggingface_analyzer.py
"""
HuggingFace Modelle Analyzer (BART, RoBERTa, SiEBERT)
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
import time
import numpy as np

from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from config.emotion_mappings import EKMAN_EMOTIONS, get_all_emotion_terms

class HuggingFaceAnalyzer(BaseAnalyzer):
    """HuggingFace Modelle Analyzer"""
    
    def __init__(self, model_name: str, api_config):
        super().__init__(model_name, api_config)
        self.pipeline = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialisiert das HuggingFace Modell"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            
            if self.model_name == "facebook/bart-large":
                self.pipeline = pipeline(
                    task="fill-mask",
                    model=self.model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=device
                )
            elif self.model_name == "FacebookAI/roberta-base":
                self.pipeline = pipeline(
                    task="fill-mask",
                    model=self.model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=device
                )
            elif self.model_name == "siebert/sentiment-roberta-large-english":
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=device
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing model {self.model_name}: {e}")
            self.pipeline = None
    
    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        """Analysiert einen einzelnen Text"""
        try:
            start_time = time.time()
            
            if analysis_type == "valence":
                scores = self._analyze_valence(text, **kwargs)
            elif analysis_type == "ekman":
                scores = self._analyze_ekman(text, **kwargs)
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
        """Analysiert eine Liste von Texten"""
        results = []
        for text in texts:
            result = self.analyze_single(text, analysis_type, **kwargs)
            results.append(result)
        return results
    
    def is_available(self) -> bool:
        """Prüft, ob der Analyzer verfügbar ist"""
        return self.pipeline is not None
    
    def _analyze_valence(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Valence mit HuggingFace Modellen"""
        if self.model_name == "siebert/sentiment-roberta-large-english":
            # SiEBERT: Direktes Sentiment
            result = self.pipeline(text)
            if isinstance(result, list) and len(result) > 0:
                sentiment = result[0]
                if sentiment['label'] == 'POSITIVE':
                    return {
                        "positive": sentiment['score'],
                        "negative": 1 - sentiment['score'],
                        "neutral": 0.1
                    }
                else:
                    return {
                        "positive": 1 - sentiment['score'],
                        "negative": sentiment['score'],
                        "neutral": 0.1
                    }
        
        else:
            # BART/RoBERTa: Fill-Mask für Valence
            return self._fill_mask_analysis(text, ["positive", "negative", "neutral"])
    
    def _analyze_ekman(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Ekman-Emotionen mit Fill-Mask"""
        if self.model_name == "siebert/sentiment-roberta-large-english":
            # SiEBERT unterstützt nur Valence
            return {emotion: 0.1 for emotion in EKMAN_EMOTIONS.keys()}
        
        emotions = list(EKMAN_EMOTIONS.keys())
        return self._fill_mask_analysis(text, emotions)
    
    def _analyze_happiness(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Happiness für Emotion Arc"""
        if self.model_name == "siebert/sentiment-roberta-large-english":
            result = self.pipeline(text)
            if isinstance(result, list) and len(result) > 0:
                sentiment = result[0]
                if sentiment['label'] == 'POSITIVE':
                    return {"happiness": sentiment['score']}
                else:
                    return {"happiness": 1 - sentiment['score']}
        
        return self._fill_mask_analysis(text, ["happiness"])
    
    def _fill_mask_analysis(self, text: str, target_emotions: List[str]) -> Dict[str, float]:
        """Führt Fill-Mask Analyse für Emotionen durch"""
        try:
            # Template für Fill-Mask
            masked_text = f"This text makes me feel <mask>. {text}"
            
            # Hole alle möglichen Begriffe für die Ziel-Emotionen
            emotion_terms = {}
            for emotion in target_emotions:
                emotion_terms[emotion] = get_all_emotion_terms(emotion)
            
            # Führe Fill-Mask durch
            predictions = self.pipeline(masked_text)
            
            # Aggregiere Scores für jede Emotion
            emotion_scores = {emotion: 0.0 for emotion in target_emotions}
            
            for pred in predictions[:50]:  # Top 50 Vorhersagen
                token = pred['token_str'].strip().lower()
                score = pred['score']
                
                # Prüfe ob Token zu einer unserer Emotionen gehört
                for emotion, terms in emotion_terms.items():
                    if token in [term.lower() for term in terms]:
                        emotion_scores[emotion] += score
            
            # Normalisiere Scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
            else:
                # Fallback: Gleichverteilung
                emotion_scores = {k: 1.0 / len(target_emotions) for k in target_emotions}
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Error in fill-mask analysis: {e}")
            # Fallback: Gleichverteilung
            return {emotion: 1.0 / len(target_emotions) for emotion in target_emotions}
