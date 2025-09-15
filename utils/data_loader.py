# utils/data_loader.py
"""
Daten-Loader für verschiedene Input-Formate
"""

import pandas as pd
import streamlit as st
from typing import List, Tuple, Optional
import io
import chardet

class DataLoader:
    """Lädt und verarbeitet verschiedene Input-Formate"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.txt', '.xlsx']
    
    def load_from_file(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Lädt Texte aus hochgeladener Datei"""
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension == 'txt':
                return self._load_txt(uploaded_file)
            elif file_extension == 'xlsx':
                return self._load_excel(uploaded_file)
            else:
                return [], f"Nicht unterstütztes Dateiformat: {file_extension}"
                
        except Exception as e:
            return [], f"Fehler beim Laden der Datei: {str(e)}"
    
    def _load_csv(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Lädt CSV-Datei"""
        try:
            # Encoding erkennen
            raw_data = uploaded_file.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            uploaded_file.seek(0)
            
            # CSV laden
            df = pd.read_csv(uploaded_file, encoding=encoding)
            
            # Text-Spalte finden
            text_column = self._find_text_column(df)
            if not text_column:
                return [], "Keine Text-Spalte in CSV gefunden"
            
            texts = df[text_column].dropna().astype(str).tolist()
            return texts, None
            
        except Exception as e:
            return [], f"Fehler beim CSV-Import: {str(e)}"
    
    def _load_txt(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Lädt TXT-Datei"""
        try:
            # Encoding erkennen
            raw_data = uploaded_file.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            # Text dekodieren
            text_content = raw_data.decode(encoding)
            
            # Text in Zeilen aufteilen (jede Zeile = ein Text)
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            return lines, None
            
        except Exception as e:
            return [], f"Fehler beim TXT-Import: {str(e)}"
    
    def _load_excel(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Lädt Excel-Datei"""
        try:
            df = pd.read_excel(uploaded_file)
            
            # Text-Spalte finden
            text_column = self._find_text_column(df)
            if not text_column:
                return [], "Keine Text-Spalte in Excel gefunden"
            
            texts = df[text_column].dropna().astype(str).tolist()
            return texts, None
            
        except Exception as e:
            return [], f"Fehler beim Excel-Import: {str(e)}"
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Findet die wahrscheinlichste Text-Spalte"""
        text_indicators = ['text', 'content', 'message', 'comment', 'review', 'description']
        
        # Exakte Übereinstimmung
        for col in df.columns:
            if col.lower() in text_indicators:
                return col
        
        # Teilweise Übereinstimmung
        for col in df.columns:
            for indicator in text_indicators:
                if indicator in col.lower():
                    return col
        
        # Längste durchschnittliche Textlänge
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                text_columns.append((col, avg_length))
        
        if text_columns:
            text_columns.sort(key=lambda x: x[1], reverse=True)
            return text_columns[0][0]
        
        return None
    
    def validate_texts(self, texts: List[str]) -> Tuple[List[str], List[str]]:
        """Validiert und bereinigt Texte"""
        valid_texts = []
        errors = []
        
        for i, text in enumerate(texts):
            if not text or len(text.strip()) < 3:
                errors.append(f"Text {i+1}: Zu kurz (< 3 Zeichen)")
                continue
            
            if len(text) > 10000:
                errors.append(f"Text {i+1}: Zu lang (> 10.000 Zeichen)")
                continue
            
            valid_texts.append(text.strip())
        
        return valid_texts, errors
