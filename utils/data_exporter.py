# utils/data_exporter.py
"""
Export-Funktionen für Analyseergebnisse
"""

import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import io
import json
from datetime import datetime

from analyzers.base_analyzer import AnalysisResult

class DataExporter:
    """Exportiert Analyseergebnisse in verschiedene Formate"""
    
    def __init__(self):
        pass
    
    def results_to_dataframe(self, results: List[Dict[str, AnalysisResult]], analysis_type: str) -> pd.DataFrame:
        """Konvertiert Analyseergebnisse zu DataFrame"""
        rows = []
        
        for i, text_results in enumerate(results):
            base_row = {
                'text_id': i + 1,
                'text': '',
                'text_length': 0,
                'analysis_type': analysis_type
            }
            
            # Hole Text aus dem ersten verfügbaren Ergebnis
            for model_name, result in text_results.items():
                if result.text:
                    base_row['text'] = result.text
                    base_row['text_length'] = len(result.text)
                    break
            
            # Erstelle Spalten für jedes Modell
            for model_name, result in text_results.items():
                row = base_row.copy()
                row['model'] = model_name
                row['processing_time'] = result.processing_time
                row['error'] = result.error or ''
                
                # Füge Scores hinzu
                if analysis_type == 'valence':
                    row.update({
                        'positive': result.scores.get('positive', 0.0),
                        'negative': result.scores.get('negative', 0.0),
                        'neutral': result.scores.get('neutral', 0.0)
                    })
                elif analysis_type == 'ekman':
                    row.update({
                        'joy': result.scores.get('joy', 0.0),
                        'surprise': result.scores.get('surprise', 0.0),
                        'fear': result.scores.get('fear', 0.0),
                        'anger': result.scores.get('anger', 0.0),
                        'disgust': result.scores.get('disgust', 0.0),
                        'sadness': result.scores.get('sadness', 0.0),
                        'contempt': result.scores.get('contempt', 0.0)
                    })
                elif analysis_type == 'emotion_arc':
                    row.update({
                        'happiness': result.scores.get('happiness', 0.0)
                    })
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def arc_to_dataframe(self, arc_data: Dict[str, Any]) -> pd.DataFrame:
        """Konvertiert Emotion Arc Daten zu DataFrame"""
        if "error" in arc_data:
            return pd.DataFrame()
        
        segments = arc_data.get("segments", [])
        happiness_scores = arc_data.get("happiness_scores", [])
        analysis = arc_data.get("arc_analysis", {})
        
        rows = []
        for i, (segment, happiness) in enumerate(zip(segments, happiness_scores)):
            rows.append({
                'segment_id': i + 1,
                'segment_text': segment,
                'happiness_score': happiness,
                'model_used': arc_data.get("model_used", ""),
                'archetype': analysis.get("archetype", ""),
                'confidence': analysis.get("confidence", 0.0)
            })
        
        df = pd.DataFrame(rows)
        
        # Füge Key Moments als separate Spalten hinzu
        key_moments = analysis.get("key_moments", [])
        for moment in key_moments:
            df.loc[df['segment_id'] == moment['position'] + 1, 'key_moment'] = moment['type']
        
        return df
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """Exportiert DataFrame zu CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.csv"
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        return csv_buffer.getvalue()
    
    def export_to_excel(self, df: pd.DataFrame, filename: str = None) -> bytes:
        """Exportiert DataFrame zu Excel"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.xlsx"
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
        
        return excel_buffer.getvalue()
    
    def export_to_json(self, results: List[Dict[str, AnalysisResult]], filename: str = None) -> str:
        """Exportiert Ergebnisse zu JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.json"
        
        # Konvertiere AnalysisResult Objekte zu Dictionaries
        json_data = []
        for i, text_results in enumerate(results):
            text_data = {
                'text_id': i + 1,
                'models': {}
            }
            
            for model_name, result in text_results.items():
                text_data['models'][model_name] = {
                    'text': result.text,
                    'model': result.model,
                    'analysis_type': result.analysis_type,
                    'scores': result.scores,
                    'processing_time': result.processing_time,
                    'metadata': result.metadata,
                    'error': result.error
                }
            
            json_data.append(text_data)
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    def create_download_button(self, data: Any, filename: str, mime_type: str, label: str):
        """Erstellt Streamlit Download-Button"""
        return st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime_type
        )
