# models/emotion_arc_analyzer.py
"""
Emotion Arc Analyzer - Happiness Tracking über Textverlauf
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor, as_completed

from analyzers.base_analyzer import AnalysisResult
from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.vader_analyzer import VADERAnalyzer
from utils.api_manager import APIManager
from utils.text_processor import TextProcessor

class EmotionArcAnalyzer:
    """Analysiert emotionale Bögen (Happiness-Tracking) über Textverlauf"""
    
    # Story Archetypes basierend auf Reagan et al.
    STORY_ARCHETYPES = {
        'rags_to_riches': {
            'name': 'Rags to Riches',
            'description': 'Aufstieg vom Unglück zum Glück',
            'pattern': 'monotonic_rise'
        },
        'tragedy': {
            'name': 'Tragedy', 
            'description': 'Fall vom Glück ins Unglück',
            'pattern': 'monotonic_fall'
        },
        'man_in_hole': {
            'name': 'Man in a Hole',
            'description': 'Glück → Unglück → Glück',
            'pattern': 'valley'
        },
        'icarus': {
            'name': 'Icarus',
            'description': 'Unglück → Glück → Unglück', 
            'pattern': 'peak'
        },
        'cinderella': {
            'name': 'Cinderella',
            'description': 'Komplex mit mehreren Wendungen (aufwärts)',
            'pattern': 'rise_fall_rise'
        },
        'oedipus': {
            'name': 'Oedipus',
            'description': 'Komplex mit mehreren Wendungen (abwärts)',
            'pattern': 'fall_rise_fall'
        }
    }
    
    def __init__(self):
        self.api_manager = APIManager()
        self.text_processor = TextProcessor()
        self.analyzers = {}
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialisiert verfügbare Analyzer für Emotion Arc"""
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
            # SiEBERT für schnelle Happiness-Analyse
            config = self.api_manager.get_api_config("huggingface")
            if config.primary_key:
                self.analyzers["siebert/sentiment-roberta-large-english"] = HuggingFaceAnalyzer(
                    "siebert/sentiment-roberta-large-english", config
                )
        except Exception:
            pass
        
        # VADER (immer verfügbar)
        self.analyzers["vader"] = VADERAnalyzer()
    
    def analyze_arc(self, text: str, model: str = "apt-5-nano", n_segments: int = 20, **kwargs) -> Dict[str, Any]:
        """Analysiert den emotionalen Bogen eines Textes"""
        
        # Text in Segmente aufteilen
        segments = self.text_processor.extract_segments_for_arc(text, n_segments)
        
        if not segments:
            return {"error": "Konnte Text nicht segmentieren"}
        
        # Happiness-Scores für alle Segmente berechnen
        happiness_scores = []
        
        if model not in self.analyzers:
            model = "vader"  # Fallback
        
        analyzer = self.analyzers[model]
        
        if not analyzer.is_available():
            return {"error": f"Modell {model} nicht verfügbar"}
        
        # Parallele Verarbeitung der Segmente
        with ThreadPoolExecutor(max_workers=min(5, len(segments))) as executor:
            futures = {}
            
            for i, segment in enumerate(segments):
                future = executor.submit(analyzer.analyze_single, segment, "emotion_arc", **kwargs)
                futures[future] = i
            
            # Ergebnisse sammeln
            segment_scores = [0.0] * len(segments)
            
            for future in as_completed(futures):
                segment_idx = futures[future]
                try:
                    result = future.result(timeout=30)
                    if not result.error and "happiness" in result.scores:
                        segment_scores[segment_idx] = result.scores["happiness"]
                    else:
                        segment_scores[segment_idx] = 0.5  # Neutral fallback
                except Exception:
                    segment_scores[segment_idx] = 0.5  # Neutral fallback
        
        happiness_scores = np.array(segment_scores)
        
        # Arc-Analyse durchführen
        arc_data = self._analyze_emotional_arc(happiness_scores, segments)
        
        return {
            "segments": segments,
            "happiness_scores": happiness_scores.tolist(),
            "arc_analysis": arc_data,
            "model_used": model,
            "n_segments": len(segments)
        }
    
    def _analyze_emotional_arc(self, happiness_scores: np.ndarray, segments: List[str]) -> Dict[str, Any]:
        """Analysiert die emotionale Arc und klassifiziert das Muster"""
        
        # Glätte die Happiness-Kurve
        if len(happiness_scores) >= 5:
            window_length = min(len(happiness_scores) // 2, 7)
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(3, window_length)
            
            try:
                smoothed_scores = savgol_filter(happiness_scores, window_length, 2)
            except:
                smoothed_scores = happiness_scores
        else:
            smoothed_scores = happiness_scores
        
        # Normalisiere Scores auf 0-1
        min_score, max_score = smoothed_scores.min(), smoothed_scores.max()
        if max_score - min_score > 0:
            normalized_scores = (smoothed_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.full_like(smoothed_scores, 0.5)
        
        # Berechne Features
        features = self._extract_arc_features(normalized_scores)
        
        # Klassifiziere Muster
        archetype, confidence = self._classify_archetype(normalized_scores, features)
        
        # Finde Schlüsselmomente
        key_moments = self._find_key_moments(normalized_scores, segments)
        
        return {
            "raw_scores": happiness_scores.tolist(),
            "smoothed_scores": smoothed_scores.tolist(),
            "normalized_scores": normalized_scores.tolist(),
            "features": features,
            "archetype": archetype,
            "confidence": confidence,
            "key_moments": key_moments
        }
    
    def _extract_arc_features(self, scores: np.ndarray) -> Dict[str, Any]:
        """Extrahiert Features aus dem emotionalen Bogen"""
        if len(scores) < 3:
            return {"error": "Zu wenige Datenpunkte"}
        
        x = np.arange(len(scores))
        
        # Trend-Analyse
        slope, _, r_value, _, _ = linregress(x, scores)
        
        # Peaks und Valleys
        peaks, _ = find_peaks(scores, prominence=0.15)
        valleys, _ = find_peaks(-scores, prominence=0.15)
        
        # Volatilität
        volatility = np.std(np.diff(scores))
        
        # Start- und End-Position
        start_pos = self._classify_position(scores[:max(1, len(scores)//5)].mean())
        end_pos = self._classify_position(scores[-max(1, len(scores)//5):].mean())
        
        return {
            "trend": "positive" if slope > 0.01 else "negative" if slope < -0.01 else "neutral",
            "slope": float(slope),
            "r_squared": float(r_value**2),
            "n_peaks": len(peaks),
            "n_valleys": len(valleys),
            "volatility": float(volatility),
            "start_position": start_pos,
            "end_position": end_pos,
            "peak_positions": peaks.tolist(),
            "valley_positions": valleys.tolist()
        }
    
    def _classify_position(self, value: float) -> str:
        """Klassifiziert Position als low/medium/high"""
        if value < 0.33:
            return "low"
        elif value < 0.67:
            return "medium" 
        else:
            return "high"
    
    def _classify_archetype(self, scores: np.ndarray, features: Dict[str, Any]) -> Tuple[str, float]:
        """Klassifiziert den Archetyp basierend auf Features"""
        if "error" in features:
            return "unknown", 0.0
        
        archetype_scores = {}
        
        # Rags to Riches: Monotoner Anstieg
        if (features["trend"] == "positive" and 
            features["n_peaks"] + features["n_valleys"] <= 1 and
            features["start_position"] in ["low", "medium"] and
            features["end_position"] == "high"):
            archetype_scores["rags_to_riches"] = 0.8 + 0.2 * features["r_squared"]
        
        # Tragedy: Monotoner Abstieg
        if (features["trend"] == "negative" and
            features["n_peaks"] + features["n_valleys"] <= 1 and
            features["start_position"] == "high" and
            features["end_position"] in ["low", "medium"]):
            archetype_scores["tragedy"] = 0.8 + 0.2 * features["r_squared"]
        
        # Man in Hole: Ein Valley
        if (features["n_valleys"] == 1 and features["n_peaks"] <= 1 and
            features["start_position"] != "low" and features["end_position"] != "low"):
            archetype_scores["man_in_hole"] = 0.7
        
        # Icarus: Ein Peak
        if (features["n_peaks"] == 1 and features["n_valleys"] <= 1 and
            features["start_position"] != "high" and features["end_position"] != "high"):
            archetype_scores["icarus"] = 0.7
        
        # Cinderella: Mehrere Wendungen, positiver Trend
        if (features["n_peaks"] + features["n_valleys"] >= 2 and
            features["trend"] == "positive" and features["end_position"] == "high"):
            archetype_scores["cinderella"] = 0.6
        
        # Oedipus: Mehrere Wendungen, negativer Trend
        if (features["n_peaks"] + features["n_valleys"] >= 2 and
            features["trend"] == "negative" and features["end_position"] == "low"):
            archetype_scores["oedipus"] = 0.6
        
        if archetype_scores:
            best_archetype = max(archetype_scores, key=archetype_scores.get)
            confidence = archetype_scores[best_archetype]
            return best_archetype, confidence
        else:
            return "unknown", 0.0
    
    def _find_key_moments(self, scores: np.ndarray, segments: List[str]) -> List[Dict[str, Any]]:
        """Findet Schlüsselmomente (Peaks und Valleys) im Text"""
        moments = []
        
        peaks, _ = find_peaks(scores, prominence=0.1)
        valleys, _ = find_peaks(-scores, prominence=0.1)
        
        for peak_idx in peaks:
            if peak_idx < len(segments):
                moments.append({
                    "type": "peak",
                    "position": int(peak_idx),
                    "happiness": float(scores[peak_idx]),
                    "text_preview": segments[peak_idx][:100] + "..." if len(segments[peak_idx]) > 100 else segments[peak_idx]
                })
        
        for valley_idx in valleys:
            if valley_idx < len(segments):
                moments.append({
                    "type": "valley", 
                    "position": int(valley_idx),
                    "happiness": float(scores[valley_idx]),
                    "text_preview": segments[valley_idx][:100] + "..." if len(segments[valley_idx]) > 100 else segments[valley_idx]
                })
        
        # Sortiere nach Position
        moments.sort(key=lambda x: x["position"])
        
        return moments
    
    def create_arc_visualization(self, arc_data: Dict[str, Any]) -> go.Figure:
        """Erstellt Plotly-Visualisierung des emotionalen Bogens"""
        if "error" in arc_data:
            return go.Figure().add_annotation(text="Fehler bei der Arc-Analyse", xref="paper", yref="paper", x=0.5, y=0.5)
        
        analysis = arc_data["arc_analysis"]
        
        fig = go.Figure()
        
        # Raw scores (transparent)
        fig.add_trace(go.Scatter(
            x=list(range(len(analysis["raw_scores"]))),
            y=analysis["raw_scores"],
            mode='lines+markers',
            name='Raw Happiness',
            line=dict(color='lightblue', width=1),
            marker=dict(size=4),
            opacity=0.5
        ))
        
        # Smoothed scores (hauptlinie)
        fig.add_trace(go.Scatter(
            x=list(range(len(analysis["smoothed_scores"]))),
            y=analysis["smoothed_scores"],
            mode='lines',
            name='Emotional Arc',
            line=dict(color='blue', width=3)
        ))
        
        # Key moments markieren
        for moment in analysis["key_moments"]:
            fig.add_annotation(
                x=moment["position"],
                y=moment["happiness"],
                text="📈" if moment["type"] == "peak" else "📉",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red" if moment["type"] == "valley" else "green",
                ax=0,
                ay=-30 if moment["type"] == "peak" else 30
            )
        
        # Titel mit Archetyp
        archetype_info = ""
        if analysis["archetype"] != "unknown":
            archetype_name = self.STORY_ARCHETYPES[analysis["archetype"]]["name"]
            confidence = analysis["confidence"] * 100
            archetype_info = f" - {archetype_name} ({confidence:.0f}% Konfidenz)"
        
        fig.update_layout(
            title=f"Emotionaler Bogen{archetype_info}",
            xaxis_title="Text-Progression →",
            yaxis_title="Happiness Level",
            yaxis=dict(range=[0, 1]),
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def get_available_models(self) -> List[str]:
        """Gibt verfügbare Modelle für Emotion Arc zurück"""
        return [name for name, analyzer in self.analyzers.items() if analyzer.is_available()]
