# ui/main_content.py
"""
Hauptinhalt-Komponenten für Streamlit
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import time

from config.languages import get_text
from utils.data_loader import DataLoader
from utils.text_processor import TextProcessor

class MainContentUI:
    """Verwaltet den Hauptinhalt der Anwendung"""
    
    def __init__(self, language: str = "DE"):
        self.language = language
        self.data_loader = DataLoader()
        self.text_processor = TextProcessor()
    
    def render_header(self):
        """Rendert den Hauptheader"""
        st.title(get_text("title", self.language))
        st.markdown(get_text("subtitle", self.language))
        st.divider()
    
    def render_input_section(self) -> Dict[str, Any]:
        """Rendert Input-Sektion und gibt Eingabedaten zurück"""
        st.subheader("📝 " + get_text("input_method", self.language))
        
        # Input-Methode auswählen
        input_method = st.radio(
            get_text("input_method", self.language),
            options=["single_text", "batch_upload"],
            format_func=lambda x: get_text(x, self.language),
            horizontal=True,
            key="input_method"
        )
        
        if input_method == "single_text":
            return self._render_single_text_input()
        else:
            return self._render_batch_upload()
    
    def _render_single_text_input(self) -> Dict[str, Any]:
        """Rendert Einzeltext-Eingabe"""
        st.markdown("#### " + get_text("text_input", self.language))
        
        text = st.text_area(
            get_text("text_input", self.language),
            placeholder=get_text("text_placeholder", self.language),
            height=200,
            key="single_text_input",
            label_visibility="collapsed"
        )
        
        # Text-Validierung
        if text:
            is_valid, error_msg = self.text_processor.validate_text(text)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return {"texts": [], "valid": False, "method": "single"}
            else:
                cleaned_text = self.text_processor.clean_text(text)
                st.success(f"✅ Text bereit ({len(cleaned_text)} Zeichen)")
                return {"texts": [cleaned_text], "valid": True, "method": "single"}
        
        return {"texts": [], "valid": False, "method": "single"}
    
    def _render_batch_upload(self) -> Dict[str, Any]:
        """Rendert Batch-Upload"""
        st.markdown("#### " + get_text("file_upload", self.language))
        
        uploaded_file = st.file_uploader(
            get_text("file_upload", self.language),
            type=['csv', 'txt', 'xlsx'],
            help=get_text("file_types", self.language),
            key="batch_file_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            with st.spinner("Datei wird verarbeitet..."):
                texts, error = self.data_loader.load_from_file(uploaded_file)
                
                if error:
                    st.error(f"❌ {error}")
                    return {"texts": [], "valid": False, "method": "batch"}
                
                if texts:
                    # Validiere Texte
                    valid_texts, errors = self.data_loader.validate_texts(texts)
                    
                    if errors:
                        with st.expander("⚠️ Warnungen", expanded=False):
                            for error in errors:
                                st.warning(error)
                    
                    if valid_texts:
                        st.success(f"✅ {len(valid_texts)} Texte erfolgreich geladen")
                        
                        # Vorschau der ersten Texte
                        with st.expander("👁️ Vorschau", expanded=False):
                            for i, text in enumerate(valid_texts[:3]):
                                st.text(f"Text {i+1}: {text[:200]}...")
                            if len(valid_texts) > 3:
                                st.info(f"... und {len(valid_texts) - 3} weitere Texte")
                        
                        return {"texts": valid_texts, "valid": True, "method": "batch"}
                    else:
                        st.error("❌ Keine gültigen Texte gefunden")
                else:
                    st.error("❌ Keine Texte in der Datei gefunden")
        
        return {"texts": [], "valid": False, "method": "batch"}
    
    def render_analysis_button(self, input_data: Dict[str, Any], settings: Dict[str, Any]) -> bool:
        """Rendert Analyse-Button und gibt zurück ob Analyse gestartet werden soll"""
        st.divider()
        
        # Zusammenfassung anzeigen
        if input_data["valid"] and input_data["texts"]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    get_text("text_count", self.language),
                    len(input_data["texts"])
                )
            
            with col2:
                model_count = len(settings.get("selected_models", []))
                st.metric("Modelle", model_count)
            
            with col3:
                analysis_type = settings.get("analysis_type", "valence")
                st.metric("Analyse-Typ", analysis_type.title())
            
            # Analyse-Button
            button_disabled = (
                not input_data["valid"] or 
                not input_data["texts"] or 
                not settings.get("selected_models")
            )
            
            if st.button(
                get_text("analyze_button", self.language),
                type="primary",
                disabled=button_disabled,
                use_container_width=True,
                key="analyze_button"
            ):
                return True
        
        else:
            st.info("💡 " + get_text("error_no_text", self.language))
        
        return False
    
    def render_progress_section(self, total_tasks: int) -> st.progress:
        """Rendert Progress-Sektion"""
        st.divider()
        st.subheader("⚡ " + get_text("analyzing", self.language))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        return progress_bar, status_text
    
    def update_progress(self, progress_bar, status_text, current: int, total: int, task_description: str = ""):
        """Aktualisiert Progress-Anzeige"""
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        
        if task_description:
            status_text.text(f"{task_description} ({current}/{total})")
        else:
            status_text.text(f"Verarbeitet: {current}/{total}")
