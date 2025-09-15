# ui/results_display.py
"""
Ergebnis-Anzeige Komponenten
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import time

from analyzers.base_analyzer import AnalysisResult
from config.languages import get_text
from config.emotion_mappings import get_emotion_display_name
from utils.visualizer import SentimentVisualizer
from utils.data_exporter import DataExporter

class ResultsDisplayUI:
    """Verwaltet die Anzeige von Analyseergebnissen"""
    
    def __init__(self, language: str = "DE"):
        self.language = language
        self.visualizer = SentimentVisualizer(language)
        self.exporter = DataExporter()
    
    def render_results_section(self, results: List[Dict[str, AnalysisResult]], analysis_type: str, settings: Dict[str, Any]):
        """Rendert komplette Ergebnissektion"""
        st.divider()
        st.subheader("📊 " + get_text("results", self.language))
        
        if not results:
            st.warning("Keine Ergebnisse verfügbar")
            return
        
        # Ergebnis-Übersicht
        self._render_results_overview(results, analysis_type, settings)
        
        # Ergebnisse nach Typ anzeigen
        if analysis_type == "valence":
            self._render_valence_results(results, settings)
        elif analysis_type == "ekman":
            self._render_ekman_results(results, settings)
        elif analysis_type == "emotion_arc":
            self._render_emotion_arc_results(results, settings)
        
        # Export-Sektion
        self._render_export_section(results, analysis_type)
    
    def _render_results_overview(self, results: List[Dict[str, AnalysisResult]], analysis_type: str, settings: Dict[str, Any]):
        """Rendert Ergebnis-Übersicht"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Texte analysiert", len(results))
        
        with col2:
            model_count = len(settings.get("selected_models", []))
            st.metric("Modelle verwendet", model_count)
        
        with col3:
            # Durchschnittliche Verarbeitungszeit berechnen
            total_time = 0
            count = 0
            for text_results in results:
                for result in text_results.values():
                    if result.processing_time > 0:
                        total_time += result.processing_time
                        count += 1
            
            avg_time = total_time / count if count > 0 else 0
            st.metric("⌀ Zeit pro Text", f"{avg_time:.2f}s")
        
        with col4:
            # Fehlerrate berechnen
            error_count = 0
            total_count = 0
            for text_results in results:
                for result in text_results.values():
                    total_count += 1
                    if result.error:
                        error_count += 1
            
            error_rate = (error_count / total_count * 100) if total_count > 0 else 0
            st.metric("Fehlerrate", f"{error_rate:.1f}%")
    
    def _render_valence_results(self, results: List[Dict[str, AnalysisResult]], settings: Dict[str, Any]):
        """Rendert Valence-Ergebnisse"""
        st.markdown("### " + get_text("valence_results", self.language))
        
        if settings["benchmark_mode"]:
            # Benchmark-Ansicht
            self._render_valence_benchmark(results)
        else:
            # Einzelmodell-Ansicht
            self._render_valence_single(results)
    
    def _render_valence_benchmark(self, results: List[Dict[str, AnalysisResult]]):
        """Rendert Valence Benchmark-Ergebnisse"""
        # Für jeden Text ein Vergleich
        for i, text_results in enumerate(results):
            with st.expander(f"📄 Text {i+1}", expanded=(i < 3)):  # Erste 3 erweitert
                
                # Text anzeigen (gekürzt)
                first_result = next(iter(text_results.values()))
                if first_result.text:
                    text_preview = first_result.text[:200] + "..." if len(first_result.text) > 200 else first_result.text
                    st.text_area("Text", text_preview, height=100, disabled=True, key=f"valence_text_{i}")
                
                # Visualisierung
                fig = self.visualizer.create_valence_comparison(text_results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabelle mit Scores
                self._render_valence_table(text_results, f"valence_table_{i}")
    
    def _render_valence_single(self, results: List[Dict[str, AnalysisResult]]):
        """Rendert Valence Einzelmodell-Ergebnisse"""
        # Batch-Übersicht wenn mehr als 1 Text
        if len(results) > 1:
            fig = self.visualizer.create_batch_overview(results, "valence")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailergebnisse
        for i, text_results in enumerate(results):
            with st.expander(f"📄 Text {i+1}", expanded=(len(results) == 1)):
                first_result = next(iter(text_results.values()))
                
                if first_result.text:
                    text_preview = first_result.text[:300] + "..." if len(first_result.text) > 300 else first_result.text
                    st.text_area("Text", text_preview, height=120, disabled=True, key=f"single_valence_text_{i}")
                
                self._render_valence_table(text_results, f"single_valence_table_{i}")
    
    def _render_valence_table(self, text_results: Dict[str, AnalysisResult], key: str):
        """Rendert Valence-Tabelle"""
        table_data = []
        
        for model_name, result in text_results.items():
            if result.error:
                table_data.append({
                    "Modell": result.model,
                    "Positiv": "❌",
                    "Negativ": "❌", 
                    "Neutral": "❌",
                    "Zeit (s)": f"{result.processing_time:.2f}",
                    "Fehler": result.error[:50] + "..." if len(result.error) > 50 else result.error
                })
            else:
                table_data.append({
                    "Modell": result.model,
                    "Positiv": f"{result.scores.get('positive', 0):.3f}",
                    "Negativ": f"{result.scores.get('negative', 0):.3f}",
                    "Neutral": f"{result.scores.get('neutral', 0):.3f}",
                    "Zeit (s)": f"{result.processing_time:.2f}",
                    "Fehler": ""
                })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, key=key)
    
    def _render_ekman_results(self, results: List[Dict[str, AnalysisResult]], settings: Dict[str, Any]):
        """Rendert Ekman-Ergebnisse"""
        st.markdown("### " + get_text("ekman_results", self.language))
        
        # Ähnlich wie Valence, aber mit Radar-Charts
        if settings["benchmark_mode"]:
            for i, text_results in enumerate(results):
                with st.expander(f"📄 Text {i+1}", expanded=(i < 2)):
                    
                    first_result = next(iter(text_results.values()))
                    if first_result.text:
                        text_preview = first_result.text[:200] + "..." if len(first_result.text) > 200 else first_result.text
                        st.text_area("Text", text_preview, height=100, disabled=True, key=f"ekman_text_{i}")
                    
                    # Radar Chart
                    fig = self.visualizer.create_ekman_radar_chart(text_results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailtabelle
                    self._render_ekman_table(text_results, f"ekman_table_{i}")
        else:
            # Einzelmodell-Ansicht
            if len(results) > 1:
                fig = self.visualizer.create_batch_overview(results, "ekman")
                st.plotly_chart(fig, use_container_width=True)
            
            for i, text_results in enumerate(results):
                with st.expander(f"📄 Text {i+1}", expanded=(len(results) == 1)):
                    first_result = next(iter(text_results.values()))
                    
                    if first_result.text:
                        text_preview = first_result.text[:300] + "..." if len(first_result.text) > 300 else first_result.text
                        st.text_area("Text", text_preview, height=120, disabled=True, key=f"single_ekman_text_{i}")
                    
                    self._render_ekman_table(text_results, f"single_ekman_table_{i}")
    
    def _render_ekman_table(self, text_results: Dict[str, AnalysisResult], key: str):
        """Rendert Ekman-Tabelle"""
        emotions = ['joy', 'surprise', 'fear', 'anger', 'disgust', 'sadness', 'contempt']
        table_data = []
        
        for model_name, result in text_results.items():
            if result.error:
                row = {"Modell": result.model}
                for emotion in emotions:
                    row[get_emotion_display_name(emotion, self.language)] = "❌"
                row["Zeit (s)"] = f"{result.processing_time:.2f}"
                row["Fehler"] = result.error[:30] + "..." if len(result.error) > 30 else result.error
                table_data.append(row)
            else:
                row = {"Modell": result.model}
                for emotion in emotions:
                    row[get_emotion_display_name(emotion, self.language)] = f"{result.scores.get(emotion, 0):.3f}"
                row["Zeit (s)"] = f"{result.processing_time:.2f}"
                row["Fehler"] = ""
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, key=key)
    
    def _render_emotion_arc_results(self, results: List[Dict[str, AnalysisResult]], settings: Dict[str, Any]):
        """Rendert Emotion Arc-Ergebnisse"""
        st.markdown("### " + get_text("emotion_arc_results", self.language))
        st.info("⚠️ Emotion Arc Ergebnisse werden in einer separaten Sektion verarbeitet")
    
    def _render_export_section(self, results: List[Dict[str, AnalysisResult]], analysis_type: str):
        """Rendert Export-Sektion"""
        st.divider()
        st.subheader("💾 " + get_text("export", self.language))
        
        # DataFrame erstellen
        df = self.exporter.results_to_dataframe(results, analysis_type)
        
        if df.empty:
            st.warning("Keine Daten zum Exportieren verfügbar")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv_data = self.exporter.export_to_csv(df)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_csv = f"sentiment_analysis_{analysis_type}_{timestamp}.csv"
            
            st.download_button(
                label=get_text("export_csv", self.language),
                data=csv_data,
                file_name=filename_csv,
                mime="text/csv"
            )
        
        with col2:
            # Excel Export
            excel_data = self.exporter.export_to_excel(df)
            filename_excel = f"sentiment_analysis_{analysis_type}_{timestamp}.xlsx"
            
            st.download_button(
                label=get_text("export_excel", self.language),
                data=excel_data,
                file_name=filename_excel,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # JSON Export
            json_data = self.exporter.export_to_json(results)
            filename_json = f"sentiment_analysis_{analysis_type}_{timestamp}.json"
            
            st.download_button(
                label="Export als JSON",
                data=json_data,
                file_name=filename_json,
                mime="application/json"
            )
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
    
    def _render_advanced_settings(self, analysis_type: str) -> Dict[str, Any]:
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
            if any("apt-5-nano" in model for model in st.session_state.get('selected_models', [])):
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
    
    def _get_available_models
