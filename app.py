# app.py
"""
Hauptanwendung - Sentiment Analysis Toolkit
Streamlit App für umfassende Sentiment- und Emotionsanalyse
"""

import streamlit as st
import sys
import os
import logging
from typing import Dict, Any, List
import time
import traceback

# Pfad für lokale Imports hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import der Module
from ui.sidebar import SidebarUI
from ui.main_content import MainContentUI
from ui.results_display import ResultsDisplayUI
from models.valence_analyzer import ValenceAnalyzer
from models.ekman_analyzer import EkmanAnalyzer
from models.emotion_arc_analyzer import EmotionArcAnalyzer
from config.languages import get_text
from utils.visualizer import SentimentVisualizer

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisApp:
    """Hauptanwendungsklasse für das Sentiment Analysis Toolkit"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
        # UI-Komponenten
        self.sidebar_ui = SidebarUI()
        self.main_content_ui = None
        self.results_ui = None
        
        # Analyzer
        self.valence_analyzer = None
        self.ekman_analyzer = None
        self.emotion_arc_analyzer = None
        
        # Visualizer
        self.visualizer = None
    
    def setup_page_config(self):
        """Konfiguriert Streamlit Seitenlayout"""
        st.set_page_config(
            page_title="Sentiment Analysis Toolkit",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/sentiment-toolkit',
                'Report a bug': 'https://github.com/your-repo/sentiment-toolkit/issues',
                'About': """
                # Sentiment Analysis Toolkit
                Professionelle Sentiment- und Emotionsanalyse mit verschiedenen KI-Modellen.
                
                **Features:**
                - Valence-Analyse (Positiv/Negativ/Neutral)
                - Ekman-Emotionen mit Synonym-Clustering
                - Emotion Arc (Happiness-Tracking)
                - Benchmark-Modus für Modell-Vergleiche
                - Batch-Verarbeitung & Export
                """
            }
        )
    
    def initialize_session_state(self):
        """Initialisiert Session State Variablen"""
        if 'language' not in st.session_state:
            st.session_state.language = 'DE'
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        if 'arc_results' not in st.session_state:
            st.session_state.arc_results = {}
        
        if 'current_analysis_type' not in st.session_state:
            st.session_state.current_analysis_type = 'valence'
    
    def initialize_components(self):
        """Initialisiert alle Komponenten basierend auf aktueller Sprache"""
        language = st.session_state.get('language', 'DE')
        
        # UI-Komponenten
        self.main_content_ui = MainContentUI(language)
        self.results_ui = ResultsDisplayUI(language)
        self.visualizer = SentimentVisualizer(language)
        
        # Analyzer (Lazy Loading)
        if self.valence_analyzer is None:
            try:
                self.valence_analyzer = ValenceAnalyzer()
            except Exception as e:
                logger.error(f"Fehler beim Initialisieren des Valence Analyzers: {e}")
        
        if self.ekman_analyzer is None:
            try:
                self.ekman_analyzer = EkmanAnalyzer()
            except Exception as e:
                logger.error(f"Fehler beim Initialisieren des Ekman Analyzers: {e}")
        
        if self.emotion_arc_analyzer is None:
            try:
                self.emotion_arc_analyzer = EmotionArcAnalyzer()
            except Exception as e:
                logger.error(f"Fehler beim Initialisieren des Emotion Arc Analyzers: {e}")
    
    def run(self):
        """Hauptschleife der Anwendung"""
        try:
            # Komponenten initialisieren
            self.initialize_components()
            
            # Header rendern
            self.main_content_ui.render_header()
            
            # Sidebar rendern und Einstellungen holen
            settings = self.sidebar_ui.render()
            
            # Hauptinhalt rendern
            self.render_main_content(settings)
            
        except Exception as e:
            st.error("Ein unerwarteter Fehler ist aufgetreten:")
            st.code(traceback.format_exc())
            logger.error(f"Unerwarteter Fehler in der Hauptschleife: {e}")
    
    def render_main_content(self, settings: Dict[str, Any]):
        """Rendert den Hauptinhalt basierend auf Einstellungen"""
        # Input-Sektion
        input_data = self.main_content_ui.render_input_section()
        
        # Analyse-Button und -durchführung
        if self.main_content_ui.render_analysis_button(input_data, settings):
            self.run_analysis(input_data, settings)
        
        # Ergebnisse anzeigen wenn vorhanden
        if st.session_state.analysis_results and st.session_state.current_analysis_type != 'emotion_arc':
            self.results_ui.render_results_section(
                st.session_state.analysis_results,
                st.session_state.current_analysis_type,
                settings
            )
        
        # Emotion Arc Ergebnisse separat behandeln
        if st.session_state.arc_results and st.session_state.current_analysis_type == 'emotion_arc':
            self.render_emotion_arc_results(st.session_state.arc_results, settings)
    
    def run_analysis(self, input_data: Dict[str, Any], settings: Dict[str, Any]):
        """Führt die Sentiment-Analyse durch"""
        texts = input_data.get("texts", [])
        analysis_type = settings.get("analysis_type", "valence")
        selected_models = settings.get("selected_models", [])
        
        if not texts or not selected_models:
            st.error("Keine Texte oder Modelle ausgewählt")
            return
        
        # Session State aktualisieren
        st.session_state.current_analysis_type = analysis_type
        st.session_state.analysis_results = []
        st.session_state.arc_results = {}
        
        try:
            if analysis_type == "emotion_arc":
                self.run_emotion_arc_analysis(texts, settings)
            else:
                self.run_standard_analysis(texts, settings)
        
        except Exception as e:
            st.error(f"Fehler bei der Analyse: {str(e)}")
            logger.error(f"Fehler bei der Analyse: {e}")
            st.code(traceback.format_exc())
    
    def run_standard_analysis(self, texts: List[str], settings: Dict[str, Any]):
        """Führt Standard-Analyse (Valence oder Ekman) durch"""
        analysis_type = settings.get("analysis_type")
        selected_models = settings.get("selected_models", [])
        
        # Progress-Anzeige
        total_tasks = len(texts) * len(selected_models)
        progress_bar, status_text = self.main_content_ui.render_progress_section(total_tasks)
        
        results = []
        current_task = 0
        
        try:
            for i, text in enumerate(texts):
                text_results = {}
                
                # Analyzer basierend auf Typ auswählen
                if analysis_type == "valence":
                    analyzer = self.valence_analyzer
                elif analysis_type == "ekman":
                    analyzer = self.ekman_analyzer
                else:
                    raise ValueError(f"Unbekannter Analyse-Typ: {analysis_type}")
                
                if analyzer is None:
                    st.error(f"Analyzer für {analysis_type} nicht verfügbar")
                    return
                
                # Analyse für alle ausgewählten Modelle durchführen
                if settings.get("benchmark_mode", False):
                    # Benchmark-Modus: Alle Modelle parallel
                    text_results = analyzer.analyze_single(
                        text, 
                        models=selected_models,
                        **self._get_model_specific_kwargs(settings)
                    )
                    current_task += len(selected_models)
                else:
                    # Einzelmodell-Modus
                    model_name = selected_models[0]
                    text_results = analyzer.analyze_single(
                        text,
                        models=[model_name],
                        **self._get_model_specific_kwargs(settings)
                    )
                    current_task += 1
                
                results.append(text_results)
                
                # Progress aktualisieren
                self.main_content_ui.update_progress(
                    progress_bar, 
                    status_text, 
                    current_task, 
                    total_tasks,
                    f"Text {i+1}/{len(texts)} analysiert"
                )
        
        except Exception as e:
            st.error(f"Fehler während der Analyse: {str(e)}")
            logger.error(f"Fehler während der Standard-Analyse: {e}")
            return
        
        # Ergebnisse speichern
        st.session_state.analysis_results = results
        
        # Erfolgsmeldung
        progress_bar.progress(1.0)
        status_text.text("✅ Analyse abgeschlossen!")
        st.success(f"🎉 {len(texts)} Texte erfolgreich analysiert!")
    
    def run_emotion_arc_analysis(self, texts: List[str], settings: Dict[str, Any]):
        """Führt Emotion Arc Analyse durch"""
        if len(texts) != 1:
            st.error("Emotion Arc Analyse unterstützt nur einen Text pro Durchlauf")
            return
        
        text = texts[0]
        selected_models = settings.get("selected_models", [])
        
        if not selected_models:
            st.error("Kein Modell für Emotion Arc ausgewählt")
            return
        
        model = selected_models[0]  # Nur ein Modell für Arc
        n_segments = settings.get("n_segments", 20)
        
        # Progress-Anzeige
        progress_bar, status_text = self.main_content_ui.render_progress_section(n_segments)
        
        try:
            status_text.text("Starte Emotion Arc Analyse...")
            
            # Emotion Arc Analyse durchführen
            arc_result = self.emotion_arc_analyzer.analyze_arc(
                text=text,
                model=model,
                n_segments=n_segments,
                **self._get_model_specific_kwargs(settings)
            )
            
            if "error" in arc_result:
                st.error(f"Fehler bei der Arc-Analyse: {arc_result['error']}")
                return
            
            # Ergebnisse speichern
            st.session_state.arc_results = arc_result
            
            # Progress abschließen
            progress_bar.progress(1.0)
            status_text.text("✅ Emotion Arc Analyse abgeschlossen!")
            st.success(f"🎉 Emotional Arc für Text erfolgreich analysiert!")
            
        except Exception as e:
            st.error(f"Fehler bei der Emotion Arc Analyse: {str(e)}")
            logger.error(f"Fehler bei der Emotion Arc Analyse: {e}")
    
    def render_emotion_arc_results(self, arc_data: Dict[str, Any], settings: Dict[str, Any]):
        """Rendert Emotion Arc Ergebnisse"""
        st.divider()
        st.subheader("📈 " + get_text("emotion_arc_results", st.session_state.language))
        
        if "error" in arc_data:
            st.error(f"Fehler: {arc_data['error']}")
            return
        
        # Arc-Visualisierung
        if self.emotion_arc_analyzer:
            try:
                fig = self.emotion_arc_analyzer.create_arc_visualization(arc_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Fehler bei der Visualisierung: {str(e)}")
                logger.error(f"Fehler bei der Arc-Visualisierung: {e}")
        
        # Arc-Details in Expander
        with st.expander("📊 Arc-Analyse Details", expanded=True):
            analysis = arc_data.get("arc_analysis", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                archetype = analysis.get("archetype", "unknown")
                if archetype != "unknown" and self.emotion_arc_analyzer:
                    archetype_info = self.emotion_arc_analyzer.STORY_ARCHETYPES.get(archetype, {})
                    st.metric(
                        get_text("arc_pattern", st.session_state.language),
                        archetype_info.get("name", archetype)
                    )
                else:
                    st.metric(get_text("arc_pattern", st.session_state.language), "Unbekannt")
            
            with col2:
                confidence = analysis.get("confidence", 0.0)
                st.metric(
                    get_text("arc_confidence", st.session_state.language),
                    f"{confidence*100:.1f}%"
                )
            
            with col3:
                key_moments = analysis.get("key_moments", [])
                st.metric(
                    get_text("key_moments", st.session_state.language),
                    len(key_moments)
                )
        
        # Key Moments Details
        if analysis.get("key_moments"):
            with st.expander("🎭 " + get_text("key_moments", st.session_state.language)):
                for moment in analysis["key_moments"]:
                    moment_type = "📈 Peak" if moment["type"] == "peak" else "📉 Valley"
                    st.write(f"**{moment_type}** (Position {moment['position']+1})")
                    st.write(f"Happiness: {moment['happiness']:.3f}")
                    st.write(f"Text: *{moment['text_preview']}*")
                    st.divider()
        
        # Export für Arc-Daten
        if st.button("📊 Arc-Daten als CSV exportieren", use_container_width=True):
            try:
                from utils.data_exporter import DataExporter
                exporter = DataExporter()
                arc_df = exporter.arc_to_dataframe(arc_data)
                
                if not arc_df.empty:
                    csv_data = exporter.export_to_csv(arc_df)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"emotion_arc_{timestamp}.csv"
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                else:
                    st.error("Keine Arc-Daten zum Exportieren verfügbar")
            except Exception as e:
                st.error(f"Fehler beim Export: {str(e)}")
    
    def _get_model_specific_kwargs(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Holt modell-spezifische Parameter aus den Einstellungen"""
        kwargs = {}
        
        # OpenAI-spezifische Parameter
        if settings.get("reasoning_effort"):
            kwargs["reasoning_effort"] = settings["reasoning_effort"]
        
        if settings.get("verbosity"):
            kwargs["verbosity"] = settings["verbosity"]
        
        # Timeout
        if settings.get("timeout"):
            kwargs["timeout"] = settings["timeout"]
        
        return kwargs

def main():
    """Hauptfunktion - Entry Point der Anwendung"""
    try:
        # App initialisieren und starten
        app = SentimentAnalysisApp()
        app.run()
        
    except Exception as e:
        st.error("Kritischer Fehler beim Starten der Anwendung:")
        st.code(traceback.format_exc())
        logger.critical(f"Kritischer Fehler beim App-Start: {e}")

if __name__ == "__main__":
    main()
