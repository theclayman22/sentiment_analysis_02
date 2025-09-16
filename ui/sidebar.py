# ui/sidebar.py
"""
Streamlit Sidebar Komponenten
"""

import streamlit as st
from typing import List, Dict, Any, Optional

from config.settings import Settings
from config.languages import get_text, TRANSLATIONS

class SidebarUI:
    """Verwaltet die Sidebar-Komponenten"""
    
    def __init__(self):
        self.language = st.session_state.get('language', 'DE')
    
    def render(self) -> Dict[str, Any]:
        """Rendert die komplette Sidebar und gibt Einstellungen zurück"""
        settings = {}
        
        with st.sidebar:
            # Sprach-Einstellungen
            settings['language'] = self._render_language_selector()
            
            st.divider()
            
            # Analyse-Typ Auswahl
            settings['analysis_type'] = self._render_analysis_type_selector()
            
            st.divider()
            
            # Modell-Auswahl
            model_settings = self._render_model_selector(settings['analysis_type'])
            settings.update(model_settings)
            st.session_state['selected_models'] = model_settings.get('selected_models', [])

            st.divider()

            # Erweiterte Einstellungen
            settings.update(self._render_advanced_settings(
                settings['analysis_type'],
                model_settings.get('selected_models', [])
            ))
            
            st.divider()
            
            # Info-Bereich
            self._render_info_section()
        
        return settings
    
    def _render_language_selector(self) -> str:
        """Rendert Sprach-Auswahl"""
        st.subheader("🌐 " + get_text("language_settings", self.language))
        
        language = st.selectbox(
            "Language / Sprache",
            options=["DE", "EN"],
            index=0 if self.language == "DE" else 1,
            key="language_selector"
        )
        
        # Update session state wenn sich Sprache ändert
        if language != st.session_state.get('language', 'DE'):
            st.session_state['language'] = language
            st.rerun()
        
        return language
    
    def _render_analysis_type_selector(self) -> str:
        """Rendert Analyse-Typ Auswahl"""
        st.subheader("🎯 " + get_text("analysis_type", self.language))
        
        analysis_options = {
            "valence": get_text("valence_results", self.language),
            "ekman": get_text("ekman_results", self.language), 
            "emotion_arc": get_text("emotion_arc_results", self.language)
        }
        
        analysis_type = st.selectbox(
            get_text("analysis_type", self.language),
            options=list(analysis_options.keys()),
            format_func=lambda x: analysis_options[x],
            key="analysis_type_selector"
        )
        
        return analysis_type
    
    def _render_model_selector(self, analysis_type: str) -> Dict[str, Any]:
        """Rendert Modell-Auswahl basierend auf Analyse-Typ"""
        st.subheader("🤖 " + get_text("model_selection", self.language))
        
        # Verfügbare Modelle für Analyse-Typ filtern
        available_models = self._get_available_models_for_analysis(analysis_type)
        
        # Benchmark-Modus Option
        benchmark_mode = st.checkbox(
            get_text("benchmark_mode", self.language),
            help=get_text("benchmark_description", self.language),
            key="benchmark_mode"
        )
        
        selected_models = []
        
        if benchmark_mode:
            selected_models = available_models
            st.info(f"📊 {len(available_models)} Modelle ausgewählt für Benchmark")
        else:
            # Einzelmodell-Auswahl
            if available_models:
                selected_model = st.selectbox(
                    get_text("single_model", self.language),
                    options=available_models,
                    format_func=lambda x: Settings.MODELS[x].display_name,
                    key="single_model_selector"
                )
                selected_models = [selected_model]
            else:
                st.error("Keine Modelle für diesen Analyse-Typ verfügbar")
        
        return {
            "benchmark_mode": benchmark_mode,
            "selected_models": selected_models
        }
    
    def _render_advanced_settings(self, analysis_type: str, selected_models: List[str]) -> Dict[str, Any]:
        """Rendert erweiterte Einstellungen"""
        settings = {}

        with st.expander("⚙️ Erweiterte Einstellungen"):

            if analysis_type == "emotion_arc":
                settings['n_segments'] = st.slider(
                    "Anzahl Segmente für Arc-Analyse",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Mehr Segmente = detailliertere Analyse, aber langsamere Verarbeitung"
                )

            # OpenAI-spezifische Einstellungen
            if any("apt-5-nano" in model for model in selected_models):
                settings['reasoning_effort'] = st.selectbox(
                    "OpenAI Reasoning Effort",
                    options=["minimal", "low", "medium", "high"],
                    value="minimal",
                    help="Höhere Werte = bessere Qualität, aber langsamere Verarbeitung"
                )
                
                settings['verbosity'] = st.selectbox(
                    "OpenAI Verbosity",
                    options=["low", "medium", "high"],
                    value="low"
                )
            
            # Batch-Verarbeitung
            settings['batch_size'] = st.slider(
                "Batch-Größe",
                min_value=1,
                max_value=Settings.MAX_BATCH_SIZE,
                value=Settings.DEFAULT_BATCH_SIZE,
                help="Anzahl Texte die parallel verarbeitet werden"
            )
            
            # Timeout-Einstellungen
            settings['timeout'] = st.slider(
                "Timeout (Sekunden)",
                min_value=10,
                max_value=120,
                value=30,
                help="Maximale Wartezeit pro Anfrage"
            )
        
        return settings
    
    def _render_info_section(self):
        """Rendert Info-Bereich"""
        with st.expander("ℹ️ Information"):
            st.markdown("""
            **Sentiment Analysis Toolkit**
            
            **Verfügbare Modelle:**
            - 🧠 OpenAI GPT-5 Nano (apt-5-nano)
            - 🤖 DeepSeek Chat
            - 🤗 HuggingFace BART Large
            - 🤗 HuggingFace RoBERTa Base
            - 🤗 SiEBERT (Sentiment RoBERTa)
            - 📊 VADER (Klassisches Lexikon)
            
            **Analyse-Typen:**
            - **Valence**: Positiv/Negativ/Neutral
            - **Ekman**: 7 Basis-Emotionen mit Synonymen
            - **Emotion Arc**: Happiness-Tracking über Textverlauf
            
            **Features:**
            - Benchmark-Modus für Modell-Vergleiche
            - Batch-Verarbeitung für multiple Texte
            - CSV/Excel/JSON Export
            - Interaktive Visualisierungen
            """)
    
    def _get_available_models_for_analysis(self, analysis_type: str) -> List[str]:
        """Gibt verfügbare Modelle für Analyse-Typ zurück"""
        available_models = []
        
        for model_name, model_config in Settings.MODELS.items():
            if analysis_type == "valence" and model_config.supports_valence:
                available_models.append(model_name)
            elif analysis_type == "ekman" and model_config.supports_ekman:
                available_models.append(model_name)
            elif analysis_type == "emotion_arc" and model_config.supports_emotion_arc:
                available_models.append(model_name)
        
        return available_models
