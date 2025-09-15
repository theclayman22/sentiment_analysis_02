# utils/visualizer.py
"""
Visualisierungs-Utilities für Sentiment-Analyse Ergebnisse
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from analyzers.base_analyzer import AnalysisResult
from config.emotion_mappings import get_emotion_display_name

class SentimentVisualizer:
    """Erstellt Visualisierungen für Sentiment-Analyse Ergebnisse"""
    
    def __init__(self, language: str = "DE"):
        self.language = language
        self.colors = {
            'positive': '#2E8B57',
            'negative': '#DC143C', 
            'neutral': '#708090',
            'joy': '#FFD700',
            'surprise': '#FF69B4',
            'fear': '#4B0082',
            'anger': '#FF4500',
            'disgust': '#9ACD32',
            'sadness': '#4682B4',
            'contempt': '#8B4513'
        }
    
    def create_valence_comparison(self, results: Dict[str, AnalysisResult]) -> go.Figure:
        """Erstellt Balkendiagramm für Valence-Vergleich zwischen Modellen"""
        models = []
        positive_scores = []
        negative_scores = []
        neutral_scores = []
        
        for model_name, result in results.items():
            if not result.error:
                models.append(result.model)
                positive_scores.append(result.scores.get('positive', 0))
                negative_scores.append(result.scores.get('negative', 0))
                neutral_scores.append(result.scores.get('neutral', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=get_emotion_display_name('positive', self.language),
            x=models,
            y=positive_scores,
            marker_color=self.colors['positive']
        ))
        
        fig.add_trace(go.Bar(
            name=get_emotion_display_name('negative', self.language),
            x=models,
            y=negative_scores,
            marker_color=self.colors['negative']
        ))
        
        fig.add_trace(go.Bar(
            name=get_emotion_display_name('neutral', self.language),
            x=models,
            y=neutral_scores,
            marker_color=self.colors['neutral']
        ))
        
        fig.update_layout(
            title="Valence-Vergleich zwischen Modellen",
            xaxis_title="Modelle",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_ekman_radar_chart(self, results: Dict[str, AnalysisResult]) -> go.Figure:
        """Erstellt Radar-Chart für Ekman-Emotionen"""
        emotions = ['joy', 'surprise', 'fear', 'anger', 'disgust', 'sadness', 'contempt']
        emotion_labels = [get_emotion_display_name(e, self.language) for e in emotions]
        
        fig = go.Figure()
        
        for model_name, result in results.items():
            if not result.error:
                scores = [result.scores.get(emotion, 0) for emotion in emotions]
                
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=emotion_labels,
                    fill='toself',
                    name=result.model,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Ekman-Emotionen Vergleich",
            height=500
        )
        
        return fig
    
    def create_batch_overview(self, results: List[Dict[str, AnalysisResult]], analysis_type: str) -> go.Figure:
        """Erstellt Übersicht für Batch-Analyse"""
        if analysis_type == "valence":
            return self._create_valence_batch_overview(results)
        elif analysis_type == "ekman":
            return self._create_ekman_batch_overview(results)
        else:
            return go.Figure()
    
    def _create_valence_batch_overview(self, results: List[Dict[str, AnalysisResult]]) -> go.Figure:
        """Erstellt Batch-Übersicht für Valence"""
        # Sammle Daten
        text_ids = []
        positive_scores = []
        negative_scores = []
        models = []
        
        for i, text_results in enumerate(results):
            for model_name, result in text_results.items():
                if not result.error:
                    text_ids.append(i + 1)
                    positive_scores.append(result.scores.get('positive', 0))
                    negative_scores.append(result.scores.get('negative', 0))
                    models.append(result.model)
        
        df = pd.DataFrame({
            'Text ID': text_ids,
            'Positive': positive_scores,
            'Negative': negative_scores,
            'Model': models
        })
        
        # Erstelle Subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Positive Scores', 'Negative Scores'),
            shared_xaxes=True
        )
        
        # Positive Scores
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['Text ID'],
                    y=model_data['Positive'],
                    mode='lines+markers',
                    name=f'{model} (Positive)',
                    line=dict(color=self.colors['positive']),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Negative Scores
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['Text ID'],
                    y=model_data['Negative'],
                    mode='lines+markers',
                    name=f'{model} (Negative)',
                    line=dict(color=self.colors['negative']),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Valence Scores über alle Texte",
            height=600,
            xaxis_title="Text ID"
        )
        
        return fig
    
    def _create_ekman_batch_overview(self, results: List[Dict[str, AnalysisResult]]) -> go.Figure:
        """Erstellt Batch-Übersicht für Ekman-Emotionen"""
        emotions = ['joy', 'anger', 'fear', 'sadness']  # Wichtigste 4 für Übersicht
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[get_emotion_display_name(e, self.language) for e in emotions]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, emotion in enumerate(emotions):
            row, col = positions[idx]
            
            # Sammle Daten für diese Emotion
            text_ids = []
            emotion_scores = []
            models = []
            
            for i, text_results in enumerate(results):
                for model_name, result in text_results.items():
                    if not result.error:
                        text_ids.append(i + 1)
                        emotion_scores.append(result.scores.get(emotion, 0))
                        models.append(result.model)
            
            df = pd.DataFrame({
                'Text ID': text_ids,
                'Score': emotion_scores,
                'Model': models
            })
            
            # Füge Traces hinzu
            for model in df['Model'].unique():
                model_data = df[df['Model'] == model]
                fig.add_trace(
                    go.Scatter(
                        x=model_data['Text ID'],
                        y=model_data['Score'],
                        mode='lines+markers',
                        name=f'{model}',
                        line=dict(color=self.colors.get(emotion, '#000000')),
                        showlegend=(idx == 0)  # Nur bei der ersten Emotion Legend zeigen
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Ekman-Emotionen über alle Texte",
            height=600
        )
        
        return fig
    
    def create_model_performance_comparison(self, results: List[Dict[str, AnalysisResult]]) -> go.Figure:
        """Erstellt Performance-Vergleich zwischen Modellen"""
        model_times = {}
        model_errors = {}
        
        for text_results in results:
            for model_name, result in text_results.items():
                if model_name not in model_times:
                    model_times[model_name] = []
                    model_errors[model_name] = 0
                
                model_times[model_name].append(result.processing_time)
                if result.error:
                    model_errors[model_name] += 1
        
        models = list(model_times.keys())
        avg_times = [np.mean(model_times[model]) for model in models]
        error_rates = [model_errors[model] / len(results) * 100 for model in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Durchschnittliche Verarbeitungszeit', 'Fehlerrate (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Verarbeitungszeit
        fig.add_trace(
            go.Bar(
                x=models,
                y=avg_times,
                name='Verarbeitungszeit (s)',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Fehlerrate
        fig.add_trace(
            go.Bar(
                x=models,
                y=error_rates,
                name='Fehlerrate (%)',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Modell-Performance Vergleich",
            height=400,
            showlegend=False
        )
        
        return fig
