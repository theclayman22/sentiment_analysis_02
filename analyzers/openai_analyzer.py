# analyzers/openai_analyzer.py
"""
OpenAI GPT-5 Nano Analyzer mit Reasoning API
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
import time

from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from config.emotion_mappings import EKMAN_EMOTIONS, VALENCE_CATEGORIES
from utils.api_manager import APIManager

class OpenAIAnalyzer(BaseAnalyzer):
    """OpenAI GPT-5 Nano Analyzer mit Reasoning API"""
    
    def __init__(self, api_config):
        super().__init__("apt-5-nano", api_config)
        self.client = OpenAI(
            api_key=api_config.primary_key,
            base_url=api_config.base_url
        )
        
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
        try:
            # Test mit minimaler Anfrage
            response = self.client.responses.create(
                model="apt-5-nano",
                input="Test",
                reasoning={"effort": "minimal"}
            )
            return True
        except Exception:
            return False
    
    def _analyze_valence(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Valence (Positiv/Negativ/Neutral)"""
        reasoning_effort = kwargs.get("reasoning_effort", "minimal")
        
        prompt = f"""Analyze the sentiment valence of this text. Provide scores between 0.0 and 1.0 for each category.

Text: "{text}"

Return your analysis as a JSON object with these exact keys:
- "positive": score from 0.0 to 1.0
- "negative": score from 0.0 to 1.0  
- "neutral": score from 0.0 to 1.0

The scores should sum to approximately 1.0. Focus on the overall emotional tone."""

        response = self.client.responses.create(
            model="apt-5-nano",
            input=prompt,
            reasoning={"effort": reasoning_effort}
        )
        
        return self._parse_scores_response(response.output, ["positive", "negative", "neutral"])
    
    def _analyze_ekman(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Ekman-Emotionen"""
        reasoning_effort = kwargs.get("reasoning_effort", "minimal")
        
        emotions = list(EKMAN_EMOTIONS.keys())
        emotion_descriptions = []
        for emotion, data in EKMAN_EMOTIONS.items():
            synonyms = ", ".join(data["synonyms"][:5])  # Erste 5 Synonyme
            emotion_descriptions.append(f"- {emotion}: {data['name_en']} (includes: {synonyms})")
        
        emotion_list = "\n".join(emotion_descriptions)
        
        prompt = f"""Analyze the emotions in this text using the Ekman emotion model. Consider synonyms and related terms.

Text: "{text}"

Emotions to analyze:
{emotion_list}

Return your analysis as a JSON object with scores from 0.0 to 1.0 for each emotion:
- "joy": score from 0.0 to 1.0
- "surprise": score from 0.0 to 1.0
- "fear": score from 0.0 to 1.0
- "anger": score from 0.0 to 1.0
- "disgust": score from 0.0 to 1.0
- "sadness": score from 0.0 to 1.0
- "contempt": score from 0.0 to 1.0

Consider the intensity and presence of each emotion, including synonymous expressions."""

        response = self.client.responses.create(
            model="apt-5-nano",
            input=prompt,
            reasoning={"effort": reasoning_effort}
        )
        
        return self._parse_scores_response(response.output, emotions)
    
    def _analyze_happiness(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Happiness für Emotion Arc"""
        reasoning_effort = kwargs.get("reasoning_effort", "minimal")
        
        prompt = f"""Analyze the happiness level in this text segment. This is for emotional arc analysis.

Text: "{text}"

Return your analysis as a JSON object:
- "happiness": score from 0.0 to 1.0 (where 0.0 = very unhappy, 0.5 = neutral, 1.0 = very happy)

Focus on the overall emotional valence and happiness content of this text segment."""

        response = self.client.responses.create(
            model="apt-5-nano",
            input=prompt,
            reasoning={"effort": reasoning_effort}
        )
        
        return self._parse_scores_response(response.output, ["happiness"])
    
    def _parse_scores_response(self, response_text: str, expected_keys: List[str]) -> Dict[str, float]:
        """Parst die JSON-Antwort von OpenAI"""
        try:
            # Versuche JSON zu extrahieren
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            scores = json.loads(json_text)
            
            # Validiere und normalisiere Scores
            result = {}
            for key in expected_keys:
                value = scores.get(key, 0.0)
                result[key] = max(0.0, min(1.0, float(value)))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing OpenAI response: {e}")
            # Fallback: Gleiche Verteilung
            return {key: 1.0 / len(expected_keys) for key in expected_keys}
