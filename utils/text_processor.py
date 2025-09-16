# utils/text_processor.py
"""
Text-Vorverarbeitung und -Utilities
"""

import logging
import re
from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class TextProcessor:
    """Verarbeitet und bereinigt Texte für die Analyse"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._punkt_available = False
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Stellt sicher, dass NLTK-Daten verfügbar sind"""
        try:
            nltk.data.find('tokenizers/punkt')
            self._punkt_available = True
            return
        except LookupError:
            pass

        try:
            nltk.download('punkt', quiet=True, raise_on_error=False)
            nltk.data.find('tokenizers/punkt')
            self._punkt_available = True
        except LookupError:
            self.logger.warning(
                "NLTK resource 'punkt' not available. Falling back to simple sentence splitting."
            )
            self._punkt_available = False
        except Exception as exc:
            self.logger.warning(
                "Could not download NLTK resource 'punkt': %s", exc
            )
            self._punkt_available = False

    def clean_text(self, text: str) -> str:
        """Bereinigt Text für die Analyse"""
        if not text:
            return ""
        
        # Entferne übermäßige Whitespaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Entferne HTML-Tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Entferne URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Entferne E-Mail-Adressen
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Teilt Text in Sätze auf"""
        if self._punkt_available:
            try:
                sentences = sent_tokenize(text)
                return [self.clean_text(sent) for sent in sentences if sent.strip()]
            except LookupError:
                self.logger.warning(
                    "NLTK sentence tokenizer unavailable at runtime. Using fallback splitter."
                )
                self._punkt_available = False
            except Exception as exc:
                self.logger.warning(
                    "Error during sentence tokenization: %s. Falling back to simple split.",
                    exc,
                )

        # Fallback: Split by periods
        sentences = text.split('.')
        return [self.clean_text(sent) for sent in sentences if sent.strip()]
    
    def chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """Teilt langen Text in Chunks mit Überlappung"""
        words = text.split()
        
        if len(words) <= max_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks
    
    def validate_text(self, text: str, min_length: int = 3, max_length: int = 10000) -> tuple[bool, Optional[str]]:
        """Validiert Text für die Analyse"""
        if not text or not text.strip():
            return False, "Text ist leer"
        
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) < min_length:
            return False, f"Text zu kurz (minimum {min_length} Zeichen)"
        
        if len(cleaned_text) > max_length:
            return False, f"Text zu lang (maximum {max_length} Zeichen)"
        
        return True, None
    
    def extract_segments_for_arc(self, text: str, n_segments: int = 20) -> List[str]:
        """Extrahiert Segmente für Emotion Arc Analyse"""
        words = text.split()
        total_words = len(words)
        
        if total_words < n_segments:
            return [text]
        
        # Dynamische Segmentgröße mit Überlappung
        segment_size = max(20, total_words // n_segments)
        overlap = segment_size // 2
        
        segments = []
        start = 0
        
        while start < total_words and len(segments) < n_segments:
            end = min(start + segment_size, total_words)
            segment = ' '.join(words[start:end])
            segments.append(segment)
            start += segment_size - overlap
        
        return segments[:n_segments]
