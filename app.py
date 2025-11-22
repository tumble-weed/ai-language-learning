"""
Streamlit frontend for Language Learning Difficulty Analyzer
Provides audio upload, processing pipeline, and interactive results visualization.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Optional
import os
import html
import base64
import io
import math
from textwrap import dedent
import soundfile as sf

# # Import your existing modules
# from segmentation.webrtc_dialogue_segmentation import segment_dialogue
# from transcription.indic_chunker import indic_transcribe_chunks
# from sentence_alignment.align_sentences import align_sentences_to_timestamps
# from metrics.hindi_models import IndicReadabilityRH1, IndicReadabilityRH2
# from metrics.sl import SentenceLengthMetric

# Page configuration
st.set_page_config(
    page_title="Language Learning Difficulty Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'sentence_annotations' not in st.session_state:
    # Maps sentence index -> {rating: int, notes: str, timestamp: pd.Timestamp}
    st.session_state.sentence_annotations = {}
if 'audio_snippets_cache' not in st.session_state:
    st.session_state.audio_snippets_cache = {}


def ensure_results_df() -> Optional[pd.DataFrame]:
    """Return the shared results DataFrame, creating it once if needed."""
    if st.session_state.results_df is not None:
        return st.session_state.results_df
    processed = st.session_state.get('processed_results')
    if processed:
        st.session_state.results_df = pd.DataFrame(processed)
    return st.session_state.results_df


@st.cache_data(show_spinner=False)
def apply_filters_and_sort(df: pd.DataFrame, sort_by: str, sl_range: tuple, rh_range: tuple) -> pd.DataFrame:
    """Cache filtered and sorted results to avoid recomputation."""
    # Apply sorting
    if sort_by == 'Difficulty (Easy → Hard)':
        df = df.sort_values(['sl', 'rh_avg'], ascending=[True, True])
    elif sort_by == 'Difficulty (Hard → Easy)':
        df = df.sort_values(['sl', 'rh_avg'], ascending=[False, False])
    elif sort_by == 'Sentence Length':
        df = df.sort_values('sl', ascending=True)
    else:  # Audio File
        df = df.sort_values('audio_file', ascending=True)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Apply filters
    filtered_df = df[
        (df['sl'] >= sl_range[0]) & 
        (df['sl'] <= sl_range[1]) &
        (df['rh_avg'] >= rh_range[0]) & 
        (df['rh_avg'] <= rh_range[1])
    ]
    
    return filtered_df


@st.cache_data(show_spinner=False)
def build_practice_card_sections(
    sentence: str,
    translation: str,
    transliteration: str,
    audio_file: str,
    audio_html: Optional[str],
    rh_avg: float,
    rh1: float,
    rh2: float,
    sl: float,
) -> Dict[str, str]:
    """Return pre-rendered HTML sections for a practice card."""
    translation_html = html.escape(str(translation))
    sentence_html = html.escape(str(sentence))
    transliteration_html = html.escape(str(transliteration)) if transliteration else ""

    metric_grid_html = dedent(f"""
    <div class="metric-grid">
        <div class="metric">
            <span>Words</span>
            <strong>{int(sl)}</strong>
        </div>
        <div class="metric">
            <span>RH Average</span>
            <strong>{rh_avg:.1f}</strong>
        </div>
        <div class="metric">
            <span>RH1</span>
            <strong>{rh1:.1f}</strong>
        </div>
        <div class="metric">
            <span>RH2</span>
            <strong>{rh2:.1f}</strong>
        </div>
    </div>
    """).strip()

    translation_block = dedent(f"""
    <div class="translation-block">
        <div class="section-label">Translation</div>
        <p>{translation_html}</p>
    </div>
    """).strip()

    answer_content = dedent(f"""
    <div class="answer-content">
        <div class="section-label">Correct Sentence</div>
        <p>{sentence_html}</p>
    </div>
    """).strip()

    if transliteration_html:
        translit_block = dedent(f"""
        <div class="translit-content">
            <div class="section-label">Transliteration</div>
            <p>{transliteration_html}</p>
        </div>
        """).strip()
    else:
        translit_block = dedent("""
        <div class="translit-content">
            <div class="section-label">Transliteration</div>
            <div class="placeholder-note">Transliteration not provided for this sentence.</div>
        </div>
        """).strip()

    audio_path = Path(audio_file)
    if audio_html is None:
        fallback_html = "<div class='audio-missing'>Audio file not available</div>"
        if audio_path.exists():
            try:
                with audio_path.open('rb') as audio_fp:
                    audio_base64 = base64.b64encode(audio_fp.read()).decode()
                fallback_html = f"""<audio controls class='practice-audio' src='data:audio/wav;base64,{audio_base64}'></audio>"""
            except Exception:
                fallback_html = "<div class='audio-missing'>Audio file could not be loaded</div>"
        audio_html = fallback_html

    audio_box_html = dedent(f"""
    <div class="audio-box">
        <div class="section-label">Audio Playback</div>
{audio_html}
        <div class="audio-filename">{html.escape(audio_path.name if audio_path.exists() else Path(audio_file).name)}</div>
    </div>
    """).strip()

    return {
        "metric_grid": metric_grid_html,
        "translation_block": translation_block,
        "answer_content": answer_content,
        "translit_block": translit_block,
        "audio_box": audio_box_html,
    }


@st.cache_data(show_spinner=False)
def extract_audio_clip(audio_data, samplerate: int, start_time: Optional[float], end_time: Optional[float]) -> Optional[bytes]:
    """Return WAV bytes for the requested time window from already loaded audio data."""
    if audio_data is None or start_time is None or end_time is None:
        return None
    
    try:
        start_val = float(start_time)
        end_val = float(end_time)
    except (TypeError, ValueError):
        return None
    
    if end_val <= start_val:
        return None

    start_sample = max(0, int(start_val * samplerate))
    end_sample = min(len(audio_data), int(end_val * samplerate))
    if start_sample >= end_sample:
        return None

    clip = audio_data[start_sample:end_sample]
    buffer = io.BytesIO()
    sf.write(buffer, clip, samplerate, format='WAV')
    return buffer.getvalue()


def render_audio_player(row: Dict) -> None:
    """Display audio snippet using original audio file if available."""

    # Check if we have the original audio loaded in session state
    if st.session_state.audio_file_path and st.session_state.audio_file_path.exists():
        # Load audio data once if not already in session state
        if 'original_audio_data' not in st.session_state or 'original_audio_samplerate' not in st.session_state:
            try:
                data, samplerate = sf.read(str(st.session_state.audio_file_path))
                st.session_state.original_audio_data = data
                st.session_state.original_audio_samplerate = samplerate
            except Exception:
                st.session_state.original_audio_data = None
                st.session_state.original_audio_samplerate = None
        
        # Extract clip from cached audio data
        if st.session_state.original_audio_data is not None:
            start_time = row.get('start_time')
            end_time = row.get('end_time')
            
            # Create unique cache key based on start and end times
            cache_key = f"{start_time}_{end_time}"
            
            # Check if snippet is already in cache
            if cache_key not in st.session_state.audio_snippets_cache:
                # Extract and cache the snippet
                snippet = extract_audio_clip(
                    st.session_state.original_audio_data,
                    st.session_state.original_audio_samplerate,
                    start_time,
                    end_time
                )
                st.session_state.audio_snippets_cache[cache_key] = snippet
            else:
                # Load from cache
                snippet = st.session_state.audio_snippets_cache[cache_key]
            
            caption_parts = []
            if st.session_state.audio_file_path:
                caption_parts.append(st.session_state.audio_file_path.name)
            if start_time is not None and end_time is not None:
                try:
                    caption_parts.append(f"{float(start_time):.2f}s → {float(end_time):.2f}s")
                except (TypeError, ValueError):
                    pass
            caption_text = " | ".join(caption_parts) if caption_parts else "Audio clip"
            
            if snippet:
                st.audio(snippet, format='audio/wav')
                st.caption(f"📁 {caption_text}")
                return
    
    st.warning("⚠️ Audio clip not available for this sentence")


def format_timestamp(seconds: Optional[float]) -> str:
    """Return mm:ss.mmm string for display."""
    if seconds is None:
        return "--"
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(total):
        return "--"
    if total < 0:
        total = 0
    minutes = int(total // 60)
    sec_fraction = total - minutes * 60
    return f"{minutes:02d}:{sec_fraction:05.2f}"


def normalize_timestamp_units(df: pd.DataFrame) -> None:
    """Ensure start/end columns are floats in seconds (auto-detect ms)."""
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')

    start_vals = df['start_time'].dropna()
    end_vals = df['end_time'].dropna()
    needs_scaling = False
    if not start_vals.empty and start_vals.gt(1000).any():
        needs_scaling = True
    if not end_vals.empty and end_vals.gt(1000).any():
        needs_scaling = True

    if needs_scaling:
        df['start_time'] = df['start_time'] / 1000.0
        df['end_time'] = df['end_time'] / 1000.0


def save_annotation(idx: int, rating: int, notes: str, move_next: bool = False, total_sentences: int = 0):
    """Save annotation without triggering unnecessary reruns"""
    st.session_state.sentence_annotations[idx] = {
        "rating": rating,
        "notes": notes,
        "timestamp": pd.Timestamp.now()
    }
    if move_next and idx < total_sentences - 1:
        st.session_state.current_annotation_idx = idx + 1
    st.session_state.temp_rating.pop(idx, None)


def preset_save(rating_value: int, idx: int, total_sentences: int):
    """Quick save for preset buttons"""
    notes = st.session_state.get(f'notes_{idx}', '')
    if rating_value == 0:
        notes = "Needs review"
    save_annotation(idx, rating_value, notes, move_next=True, total_sentences=total_sentences)


@st.fragment
def render_annotation_form(current_idx: int, current_row: Dict, existing_annotation: Dict, total_sentences: int):
    """Isolated annotation form that reruns independently"""
    
    default_rating = existing_annotation.get('rating', 5)
    default_notes = existing_annotation.get('notes', '')
    current_rating = st.session_state.temp_rating.get(current_idx, default_rating)

    st.markdown("### 🎯 Rate Difficulty")
    st.caption("1 = Very Easy (beginner) | 10 = Very Hard (advanced)")

    st.markdown("**⚡ Quick Rate & Next:**")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("🟢 Easy (3)", use_container_width=True, key=f"preset_easy_{current_idx}"):
            preset_save(3, current_idx, total_sentences)
            st.rerun()
    with preset_cols[1]:
        if st.button("🟡 Medium (5)", use_container_width=True, key=f"preset_medium_{current_idx}"):
            preset_save(5, current_idx, total_sentences)
            st.rerun()
    with preset_cols[2]:
        if st.button("🟠 Hard (8)", use_container_width=True, key=f"preset_hard_{current_idx}"):
            preset_save(8, current_idx, total_sentences)
            st.rerun()
    with preset_cols[3]:
        if st.button("❓ Review (0)", use_container_width=True, key=f"preset_review_{current_idx}"):
            preset_save(0, current_idx, total_sentences)
            st.rerun()

    st.divider()

    with st.expander("🎚️ Fine-tune Rating (Optional)", expanded=False):
        rating_value = st.slider(
            "Detailed Rating",
            min_value=1,
            max_value=10,
            value=int(current_rating),
            step=1,
            help="Slide to adjust, then click Save",
            key=f'rating_{current_idx}'
        )
        
        # Update temp rating directly without callback
        st.session_state.temp_rating[current_idx] = rating_value

        if rating_value <= 3:
            st.success(f"🟢 Easy (Rating: {rating_value}/10)")
        elif rating_value <= 6:
            st.warning(f"🟡 Medium (Rating: {rating_value}/10)")
        elif rating_value <= 8:
            st.error(f"🟠 Hard (Rating: {rating_value}/10)")
        else:
            st.error(f"🔴 Very Hard (Rating: {rating_value}/10)")

        notes = st.text_area(
            "Notes (optional)",
            value=default_notes,
            placeholder="Add context, vocabulary issues, pronunciation notes...",
            height=100,
            key=f'notes_{current_idx}'
        )

        detail_col1, detail_col2, detail_col3 = st.columns([2, 1, 1])
        with detail_col1:
            if st.button("💾 Save & Next", type="primary", use_container_width=True, key=f"save_next_{current_idx}"):
                save_annotation(current_idx, rating_value, notes, move_next=True, total_sentences=total_sentences)
                st.rerun()
        with detail_col2:
            if st.button("💾 Save", use_container_width=True, key=f"save_{current_idx}"):
                save_annotation(current_idx, rating_value, notes, move_next=False, total_sentences=total_sentences)
                st.rerun()
        with detail_col3:
            if st.button("🗑️ Clear", use_container_width=True, key=f"clear_{current_idx}"):
                st.session_state.sentence_annotations.pop(current_idx, None)
                st.session_state.temp_rating.pop(current_idx, None)
                st.rerun()

    if current_idx in st.session_state.sentence_annotations:
        anno = st.session_state.sentence_annotations[current_idx]
        timestamp = anno.get('timestamp', 'Unknown')
        st.caption(f"✏️ Last saved: Rating {anno.get('rating')}/10 at {timestamp}")


# Title and description
st.title("🎙️ Language Learning Difficulty Analyzer")
st.markdown("""
Analyze audio recordings to identify sentence difficulty based on readability metrics and sentence length.
Upload an audio file, process it through the pipeline, and explore the results interactively.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Language selection
    lang_code = st.selectbox(
        "Select Language",
        options=['mr', 'hi', 'te', 'ta', 'kn', 'gu', 'bn', 'pa', 'or', 'as'],
        index=0,
        help="Language code for transcription (mr=Marathi, hi=Hindi, te=Telugu, etc.)"
    )
    
    # VAD settings
    st.subheader("Voice Activity Detection")
    vad_aggressiveness = st.slider(
        "VAD Aggressiveness",
        min_value=0,
        max_value=3,
        value=3,
        help="Higher values = more aggressive filtering (0-3)"
    )
    
    min_speech_len_ms = st.number_input(
        "Min Speech Length (ms)",
        min_value=100,
        max_value=2000,
        value=250,
        step=50
    )
    
    min_silence_ms = st.number_input(
        "Min Silence After Speech (ms)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50
    )
    
    # Sorting preference
    st.subheader("Results Display")
    sort_by = st.selectbox(
        "Sort Results By",
        options=['Difficulty (Easy → Hard)', 'Difficulty (Hard → Easy)', 'Sentence Length', 'Audio File']
    )

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📂 Load Results", "📤 Upload & Process", "📊 Results", "🎯 Practice", "📝 Annotate", "ℹ️ About"])

with tab1:
    st.header("Load Existing Results")
    
    st.markdown("""
    Skip the processing and directly load results from a previously generated CSV file.
    """)
    
    def load_csv_to_results(file_path_or_uploaded):
        """Load and validate CSV file into session state."""
        try:
            # Read CSV (handle both file path and uploaded file)
            if isinstance(file_path_or_uploaded, Path):
                df = pd.read_csv(file_path_or_uploaded)
            else:
                df = pd.read_csv(file_path_or_uploaded)
            
            # Column mapping for different CSV formats
            column_mapping = {
                'Audio_file': 'audio_file',
                'Text': 'sentence',
                'RH1_Result': 'rh1',
                'RH2_Result': 'rh2',
                'RH_Average': 'rh_avg',
                'SL_Result': 'sl',
                'Chunk_ID': 'audio_file',
                'WFR_Result': 'wfr',
                'Transliteration': 'transliteration',
                'Translation': 'translation',
                'Original_audio_file': 'original_audio_file',
                'Start': 'start_time',
                'End': 'end_time'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Required columns check
            required_cols = ['sentence', 'rh1', 'rh2', 'rh_avg', 'sl']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ CSV is missing required columns: {', '.join(missing_cols)}")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
                return False
            
            # Handle missing/optional columns with defaults
            if 'audio_file' not in df.columns:
                df['audio_file'] = 'N/A'
                st.warning("⚠️ CSV doesn't have audio file column. Audio playback will not be available.")
            
            for col, default in [
                ('start_time', 0),
                ('end_time', 0),
                ('wfr', None),
                ('transliteration', None),
                ('translation', None)
            ]:
                if col not in df.columns:
                    df[col] = default
            
            normalize_timestamp_units(df)
            
            # Load original audio file if available
            original_audio_path = None
            if 'original_audio_file' in df.columns and df['original_audio_file'].notna().any():
                # Get the first non-null original_audio_file path
                original_audio_path = df.loc[df['original_audio_file'].notna(), 'original_audio_file'].iloc[0]
                original_audio_path = Path(original_audio_path)
                
                if original_audio_path.exists():
                    st.session_state.audio_file_path = original_audio_path
                    
                    # Load the entire audio file into memory
                    try:
                        audio_data, samplerate = sf.read(str(original_audio_path))
                        st.session_state.original_audio_data = audio_data
                        st.session_state.original_audio_samplerate = samplerate
                        st.success(f"✅ Loaded original audio file: {original_audio_path.name}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load audio data from {original_audio_path.name}: {str(e)}")
                        st.session_state.audio_file_path = None
                        st.session_state.original_audio_data = None
                        st.session_state.original_audio_samplerate = None
                else:
                    st.warning(f"⚠️ Original audio file not found: {original_audio_path}")
                    st.session_state.audio_file_path = None
                    st.session_state.original_audio_data = None
                    st.session_state.original_audio_samplerate = None
            else:
                st.session_state.audio_file_path = None
                st.session_state.original_audio_data = None
                st.session_state.original_audio_samplerate = None
            
            # Store in session state
            st.session_state.processed_results = df.to_dict('records')
            st.session_state.results_df = df.copy()
            st.session_state.processing_complete = True
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.exception(e)
            return False
    
    # Upload CSV section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        results_file = st.file_uploader(
            "Upload Results CSV File",
            type=['csv'],
            help="Upload a CSV file with analysis results"
        )
    
    with col2:
        st.info("💡 **Quick Start**\n\nUpload a results CSV to view analysis without processing.")
    
    if results_file is not None:
        if load_csv_to_results(results_file):
            st.success(f"✅ Successfully loaded {len(st.session_state.processed_results)} sentences!")
            st.info("👉 Go to the **Results** tab to view and interact with the data")
            
            # Show preview
            st.subheader("Preview (First 5 Rows)")
            preview_df = st.session_state.results_df.head(5)[['sentence', 'rh_avg', 'sl']]
            st.dataframe(preview_df, use_container_width=True)
    
    st.divider()
    
    # Quick load from output folder
    st.subheader("📁 Quick Load from Output Folder")
    
    output_dir = Path(__file__).resolve().parent / "output"
    
    if output_dir.exists():
        csv_files = list(output_dir.glob("**/*.csv"))
        
        if csv_files:
            csv_options = {f"{f.parent.name}/{f.name}": f for f in csv_files}
            
            selected_file = st.selectbox(
                "Select a results file from output folder:",
                options=["-- Select a file --"] + list(csv_options.keys())
            )
            
            if selected_file != "-- Select a file --" and st.button("📥 Load Selected File", type="primary"):
                if load_csv_to_results(csv_options[selected_file]):
                    st.success(f"✅ Loaded {len(st.session_state.processed_results)} sentences from {csv_options[selected_file].name}!")
                    st.info("👉 Go to the **Results** tab to view the data")
        else:
            st.info("No CSV files found in output folder. Process an audio file first or upload a CSV above.")
    else:
        st.info("Output folder not found. Process an audio file first or upload a CSV above.")

with tab2:
    st.header("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, etc.)",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Upload an audio file to analyze"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.audio_file_path = Path(tmp_file.name)
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Audio player for preview
        st.audio(uploaded_file, format=f'audio/{Path(uploaded_file.name).suffix[1:]}')
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
        
        if process_button:
            st.info("⚠️ Processing pipeline commented out. Uncomment the import statements and this section to enable full processing.")
            # BASE_DIR = Path(__file__).resolve().parent
            # OUTPUT_DIR = BASE_DIR / "output" / "streamlit_temp"
            # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            # TRANSCRIBE_OUTPUT_FILE = OUTPUT_DIR / "transcribed_output.pkl"
            # SENTENCE_OUTPUT_FILE = OUTPUT_DIR / "sentences_output.txt"
            
            # progress_bar = st.progress(0)
            # status_text = st.empty()
            
            # try:
            #     # Step 1: Segmentation
            #     status_text.text("🔊 Segmenting audio into dialogue chunks...")
            #     progress_bar.progress(10)
            #     
            #     exported_chunk_paths = segment_dialogue(
            #         audio_file_path=st.session_state.audio_file_path,
            #         output_dir=OUTPUT_DIR / "chunks",
            #         vad_aggressiveness=vad_aggressiveness,
            #         min_speech_len_ms=min_speech_len_ms,
            #         min_silence_after_speech_ms=min_silence_ms
            #     )
            #     
            #     if not exported_chunk_paths:
            #         st.error("❌ No dialogue segments found. Try adjusting VAD settings.")
            #         st.stop()
            #     
            #     progress_bar.progress(30)
            #     st.info(f"Found {len(exported_chunk_paths)} dialogue segments")
            #     
            #     # Step 2: Transcription
            #     status_text.text("📝 Transcribing audio chunks...")
            #     progress_bar.progress(40)
            #     
            #     transcribed_text = indic_transcribe_chunks(
            #         lang_code=lang_code,
            #         exported_chunk_paths=exported_chunk_paths,
            #         output_file=TRANSCRIBE_OUTPUT_FILE
            #     )
            #     
            #     progress_bar.progress(60)
            #     
            #     # Step 3: Load transcription and prepare sentences
            #     status_text.text("🔄 Processing transcriptions...")
            #     with TRANSCRIBE_OUTPUT_FILE.open('rb') as f_in:
            #         transcribed_data = pickle.load(f_in)
            #     
            #     # For this simplified version, use transcribed text as sentences
            #     # (In full pipeline, you'd apply punctuation restoration and preprocessing)
            #     preprocessed_text = [transcript['text'] for transcript in transcribed_data]
            #     
            #     with SENTENCE_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
            #         f_out.write("\n".join(preprocessed_text))
            #     
            #     progress_bar.progress(70)
            #     
            #     # Step 4: Align sentences to timestamps
            #     status_text.text("⏱️ Aligning sentences to timestamps...")
            #     aligned_result = align_sentences_to_timestamps(
            #         transcribed_data,
            #         preprocessed_text,
            #         st.session_state.audio_file_path
            #     )
            #     
            #     progress_bar.progress(80)
            #     
            #     # Step 5: Calculate metrics
            #     status_text.text("📊 Calculating difficulty metrics...")
            #     rh1 = IndicReadabilityRH1()
            #     rh2 = IndicReadabilityRH2()
            #     sl = SentenceLengthMetric()
            #     
            #     results_list = []
            #     for t in aligned_result:
            #         rh1Res = rh1.compute(t["sentence"])
            #         rh2Res = rh2.compute(t["sentence"])
            #         avg = (rh1Res + rh2Res) / 2
            #         slRes = sl.compute(t["sentence"])
            #         
            #         row_data = {
            #             "audio_file": t["audio_file"],
            #             "original_audio_file": str(st.session_state.audio_file_path) if st.session_state.audio_file_path else t.get("audio_file"),
            #             "sentence": t["sentence"],
            #             "start_time": t.get("start_time", 0),
            #             "end_time": t.get("end_time", 0),
            #             "rh1": rh1Res,
            #             "rh2": rh2Res,
            #             "rh_avg": avg,
            #             "sl": slRes
            #         }
            #         results_list.append(row_data)
            #     
            #     st.session_state.processed_results = results_list
            #     st.session_state.results_df = pd.DataFrame(results_list)
            #     st.session_state.processing_complete = True
            #     
            #     progress_bar.progress(100)
            #     status_text.text("✅ Processing complete!")
            #     
            #     st.success(f"🎉 Successfully processed {len(results_list)} sentences!")
            #     st.balloons()
            #     
            # except Exception as e:
            #     st.error(f"❌ Error during processing: {str(e)}")
            #     st.exception(e)

with tab3:
    st.header("Analysis Results")
    
    if st.session_state.processing_complete and st.session_state.processed_results:
        df = ensure_results_df()
        
        # Global CSS for toggle functionality (rendered ONCE for entire tab)
        st.markdown("""
        <style>
        /* Transliteration toggle */
        .translit-toggle-checkbox { display: none; }
        .translit-toggle-btn {
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #f0f2f6;
            display: inline-block;
            user-select: none;
        }
        .translit-toggle-btn:hover { background-color: #e0e2e6; }
        .translit-icon-hide { display: none; }
        .translit-content { display: inline-block; margin-left: 10px; }
        .translit-placeholder { color: #999; }
        .translit-text { visibility: hidden; }
        .translit-toggle-checkbox:checked ~ .translit-toggle-btn .translit-icon-show { display: none; }
        .translit-toggle-checkbox:checked ~ .translit-toggle-btn .translit-icon-hide { display: inline; }
        .translit-toggle-checkbox:checked ~ .translit-content .translit-text { visibility: visible !important; }
        .translit-toggle-checkbox:checked ~ .translit-content .translit-placeholder { display: none; }
        
        /* Translation toggle */
        .trans-toggle-checkbox { display: none; }
        .trans-toggle-btn {
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #f0f2f6;
            display: inline-block;
            user-select: none;
        }
        .trans-toggle-btn:hover { background-color: #e0e2e6; }
        .trans-icon-hide { display: none; }
        .trans-content { display: inline-block; margin-left: 10px; }
        .trans-placeholder { color: #999; }
        .trans-text { visibility: hidden; }
        .trans-toggle-checkbox:checked ~ .trans-toggle-btn .trans-icon-show { display: none; }
        .trans-toggle-checkbox:checked ~ .trans-toggle-btn .trans-icon-hide { display: inline; }
        .trans-toggle-checkbox:checked ~ .trans-content .trans-text { visibility: visible !important; }
        .trans-toggle-checkbox:checked ~ .trans-content .trans-placeholder { display: none; }
        </style>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sentences", len(df))
        with col2:
            st.metric("Avg RH Score", f"{df['rh_avg'].mean():.2f}")
        with col3:
            st.metric("Avg Sentence Length", f"{df['sl'].mean():.1f} words")
        with col4:
            st.metric("Difficulty Range", f"{df['rh_avg'].min():.1f} - {df['rh_avg'].max():.1f}")
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filter Results")
        col1, col2 = st.columns(2)
        
        with col1:
            sl_range = st.slider(
                "Sentence Length Range",
                min_value=int(df['sl'].min()),
                max_value=int(df['sl'].max()),
                value=(int(df['sl'].min()), int(df['sl'].max())),
                key="results_sl_range"
            )
        
        with col2:
            rh_range = st.slider(
                "RH Average Range",
                min_value=float(df['rh_avg'].min()),
                max_value=float(df['rh_avg'].max()),
                value=(float(df['rh_avg'].min()), float(df['rh_avg'].max())),
                step=0.1,
                key="results_rh_range"
            )
        
        # Use cached filtering and sorting
        filtered_df = apply_filters_and_sort(df, sort_by, sl_range, rh_range)
        
        st.info(f"Showing {len(filtered_df)} of {len(df)} sentences")
        
        # Display layout options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📋 Sentence List")
        with col2:
            view_mode = st.radio(
                "View Mode",
                options=["Grid View", "List View"],
                horizontal=True,
                label_visibility="collapsed"
            )
        
        if view_mode == "Grid View":
            # Grid layout - 2 columns side by side
            rows = [filtered_df.iloc[i:i+2] for i in range(0, len(filtered_df), 2)]
            
            for row_idx, row_data in enumerate(rows):
                cols = st.columns(2)
                
                for col_idx, (idx, row) in enumerate(row_data.iterrows()):
                    with cols[col_idx]:
                        # Card-style container
                        with st.container():
                            st.markdown(f"##### Sentence {idx+1}")
                            st.markdown(f"**{row['sentence']}**")
                            
                            # Transliteration toggle (using global CSS)
                            if row.get('transliteration') and pd.notna(row['transliteration']):
                                st.markdown(f"""
                                <input type="checkbox" id="translit-{idx}" class="translit-toggle-checkbox">
                                <label for="translit-{idx}" class="translit-toggle-btn" title="Toggle transliteration">
                                    <span class="translit-icon-show">📝</span>
                                    <span class="translit-icon-hide">✕</span>
                                </label>
                                <div class="translit-content">
                                    <strong>Transliteration:</strong> 
                                    <span class="translit-placeholder">Show Transliteration</span>
                                    <span class="translit-text">{html.escape(row['transliteration'])}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Translation toggle (using global CSS)
                            if row.get('translation') and pd.notna(row['translation']):
                                st.markdown(f"""
                                <input type="checkbox" id="trans-{idx}" class="trans-toggle-checkbox">
                                <label for="trans-{idx}" class="trans-toggle-btn" title="Toggle translation">
                                    <span class="trans-icon-show">🌐</span>
                                    <span class="trans-icon-hide">✕</span>
                                </label>
                                <div class="trans-content">
                                    <strong>Translation:</strong> 
                                    <span class="trans-placeholder">Show Translation</span>
                                    <span class="trans-text">{html.escape(row['translation'])}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown(f"**Audio File:** `{Path(row['audio_file']).name}`")
                            
                            # Metrics in compact format
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Length", f"{row['sl']} words")
                                st.metric("RH1", f"{row['rh1']:.2f}")
                            with col2:
                                st.metric("RH2", f"{row['rh2']:.2f}")
                                st.metric("RH Avg", f"{row['rh_avg']:.2f}")
                            
                            # Audio playback
                            
                            render_audio_player(row)
                            
                            st.divider()
                            
        else:
            # List view with expanders (using global CSS)
            for idx, row in filtered_df.iterrows():
                with st.expander(f"**Sentence {idx+1}**: {row['sentence'][:50]}..." if len(row['sentence']) > 50 else f"**Sentence {idx+1}**: {row['sentence']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Full Text:** {row['sentence']}")
                        start_label = format_timestamp(row.get('start_time'))
                        end_label = format_timestamp(row.get('end_time'))
                        if start_label != "--" or end_label != "--":
                            st.caption(f"⏱️ {start_label} → {end_label}")
                        
                        # Transliteration toggle (using global CSS)
                        if row.get('transliteration') and pd.notna(row['transliteration']):
                            st.markdown(f"""
                            <input type="checkbox" id="translit-list-{idx}" class="translit-toggle-checkbox">
                            <label for="translit-list-{idx}" class="translit-toggle-btn" title="Toggle transliteration">
                                <span class="translit-icon-show">🔤</span>
                                <span class="translit-icon-hide">✕</span>
                            </label>
                            <div class="translit-content">
                                <strong>Transliteration:</strong> 
                                <span class="translit-placeholder">Show Transliteration</span>
                                <span class="translit-text">{html.escape(row['transliteration'])}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Translation toggle (using global CSS)
                        if row.get('translation') and pd.notna(row['translation']):
                            st.markdown(f"""
                            <input type="checkbox" id="trans-list-{idx}" class="trans-toggle-checkbox">
                            <label for="trans-list-{idx}" class="trans-toggle-btn" title="Toggle translation">
                                <span class="trans-icon-show">🌐</span>
                                <span class="trans-icon-hide">✕</span>
                            </label>
                            <div class="trans-content">
                                <strong>Translation:</strong> 
                                <span class="trans-placeholder">Show Translation</span>
                                <span class="trans-text">{html.escape(row['translation'])}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"**Audio File:** `{Path(row['audio_file']).name}`")
                        
                        # Audio playback if file exists
                        render_audio_player(row)
                    
                    with col2:
                        st.metric("Sentence Length", f"{row['sl']} words")
                        st.metric("RH1 Score", f"{row['rh1']:.2f}")
                        st.metric("RH2 Score", f"{row['rh2']:.2f}")
                        st.metric("RH Average", f"{row['rh_avg']:.2f}")
        
        # Download results
        st.divider()
        st.subheader("💾 Export Results")
        
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="difficulty_analysis_results.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👈 Upload and process an audio file in the **Upload & Process** tab, or load existing results in the **Load Results** tab.")

with tab4:
    st.header("🎯 Practice Mode")
    
    st.markdown("""
    Practice your language skills! See the translation and try to guess the original sentence.
    Use the controls below to tailor your practice set, then reveal the answer and audio when you're ready.
    """)
    
    if st.session_state.processing_complete and st.session_state.processed_results:
        df = ensure_results_df()
        
        # Filter to only sentences that have translations
        practice_df = df[df['translation'].notna() & (df['translation'] != '')].copy()
        
        if len(practice_df) == 0:
            st.warning("⚠️ No sentences with translations available for practice. Please load a CSV file with a Translation column.")
        else:
            # Difficulty helpers
            def categorize_difficulty(score: float) -> str:
                if score < 33:
                    return "Easy"
                if score < 66:
                    return "Medium"
                return "Hard"

            practice_df['difficulty_label'] = practice_df['rh_avg'].apply(categorize_difficulty)
            difficulty_emojis = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            difficulty_options = ["Easy", "Medium", "Hard"]

            # Enhanced Practice stats with better layout
            st.markdown("### 📊 Practice Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📚 Total Sentences", len(practice_df), help="Total available practice sentences")
            with col2:
                avg_difficulty = practice_df['rh_avg'].mean()
                st.metric("📊 Avg Difficulty", f"{avg_difficulty:.1f}", help="Average readability score")
            with col3:
                easy_count = len(practice_df[practice_df['difficulty_label'] == "Easy"])
                medium_count = len(practice_df[practice_df['difficulty_label'] == "Medium"])
                hard_count = len(practice_df[practice_df['difficulty_label'] == "Hard"])
                st.metric("🎯 Distribution", f"{easy_count}/{medium_count}/{hard_count}", help="Easy/Medium/Hard counts")
            with col4:
                avg_length = practice_df['sl'].mean()
                st.metric("📏 Avg Length", f"{avg_length:.1f} words", help="Average sentence length")
            
            st.divider()

            # Enhanced Controls with better UX
            st.markdown("### 🎛️ Practice Controls")
            
            controls_row1_col1, controls_row1_col2 = st.columns([3, 1])
            
            with controls_row1_col1:
                search_query = st.text_input(
                    "🔍 Search translation or sentence",
                    placeholder="Type keywords to filter practice cards...",
                    key="practice_search",
                    help="Search in both translations and original sentences"
                )
            
            with controls_row1_col2:
                shuffle_cards = st.checkbox("🔀 Shuffle", value=False, key="practice_shuffle", help="Randomize card order")
            
            controls_row2_col1, controls_row2_col2 = st.columns([2, 2])
            
            with controls_row2_col1:
                selected_difficulties = st.multiselect(
                    "🎚️ Difficulty Levels",
                    options=difficulty_options,
                    default=difficulty_options,
                    key="practice_difficulty",
                    help="Filter by difficulty level"
                )
            
            with controls_row2_col2:
                sl_min = int(practice_df['sl'].min())
                sl_max = int(practice_df['sl'].max())
                sentence_length_filter = st.slider(
                    "📏 Sentence Length Range",
                    min_value=sl_min,
                    max_value=sl_max,
                    value=(sl_min, sl_max),
                    key="practice_sl_filter",
                    help="Filter by number of words"
                )

            # Apply filters
            filtered_df = practice_df
            active_difficulties = selected_difficulties if selected_difficulties else difficulty_options
            filtered_df = filtered_df[filtered_df['difficulty_label'].isin(active_difficulties)]
            
            # Apply sentence length filter
            filtered_df = filtered_df[
                (filtered_df['sl'] >= sentence_length_filter[0]) & 
                (filtered_df['sl'] <= sentence_length_filter[1])
            ]

            if search_query:
                mask = (
                    filtered_df['translation'].str.contains(search_query, case=False, na=False) |
                    filtered_df['sentence'].str.contains(search_query, case=False, na=False)
                )
                filtered_df = filtered_df[mask]

            if filtered_df.empty:
                st.warning("⚠️ No sentences match your current filters. Try widening the search or selecting more difficulty levels.")
            else:
                if shuffle_cards:
                    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
                
                filtered_df = filtered_df.reset_index(drop=True)
                
                st.divider()
                
                # List view layout
                st.subheader(f"📚 Practice Cards ({len(filtered_df)} sentences)")
                
                for idx, row in filtered_df.iterrows():
                    # Card-style container
                    with st.container():
                        difficulty_label = row['difficulty_label']
                        difficulty_emoji = difficulty_emojis.get(difficulty_label, "")
                        
                        # Card header
                        st.markdown(f"##### {difficulty_emoji} Sentence #{idx+1} - {difficulty_label}")
                        
                        # Translation (always visible)
                        st.info(f"🌐 **Translation**\n\n{row['translation']}")
                        
                        # Hint
                        st.caption("💡 Try to recall the original sentence before revealing!")
                        
                        # Metrics in compact format
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Length", f"{int(row['sl'])} words")
                            st.metric("RH1", f"{row['rh1']:.2f}")
                        with col2:
                            st.metric("RH2", f"{row['rh2']:.2f}")
                            st.metric("RH Avg", f"{row['rh_avg']:.2f}")
                        
                        # Timestamps
                        start_label = format_timestamp(row.get('start_time'))
                        end_label = format_timestamp(row.get('end_time'))
                        if start_label != "--" or end_label != "--":
                            st.caption(f"⏱️ {start_label} → {end_label}")
                        
                        # Reveal answer section
                        with st.expander("👁️ Reveal Answer", expanded=False):
                            st.markdown(f"**Original Sentence:**\n\n{row['sentence']}")
                            
                            # Transliteration if available
                            if row.get('transliteration') and pd.notna(row['transliteration']):
                                st.markdown(f"**Transliteration:**\n\n{row['transliteration']}")
                            
                            # Audio playback
                            st.markdown("**Audio:**")
                            render_audio_player(row)
                            st.caption(f"📁 {Path(row['audio_file']).name}")
                        
                        st.divider()
    
    else:
        st.info("👈 Load results from the **Load Results** tab to start practicing!")

with tab5:
    st.header("📝 Difficulty Annotation")
    st.markdown("""
    Rate each sentence's difficulty on a scale of 1-10 (1=easiest, 10=hardest).
    """)

    if not (st.session_state.processing_complete and st.session_state.processed_results):
        st.warning("⚠️ Please load or process a results CSV before annotating.")
    else:
        # Initialize navigation index in session state
        if 'current_annotation_idx' not in st.session_state:
            st.session_state.current_annotation_idx = 0
        
        # Initialize temporary rating storage (prevents rerun on slider change)
        if 'temp_rating' not in st.session_state:
            st.session_state.temp_rating = {}
        
        # Use shared DataFrame reference (avoids redundant copies)
        anno_df = ensure_results_df()

        total_sentences = len(anno_df)
        annotated_count = len(st.session_state.sentence_annotations)
        
        if total_sentences == 0:
            st.info("No sentences available to annotate.")
        else:
            # Get current index
            current_idx = st.session_state.current_annotation_idx
            
            # Ensure index is within bounds
            if current_idx >= total_sentences:
                current_idx = total_sentences - 1
                st.session_state.current_annotation_idx = current_idx
            
            # Progress bar at top (static, doesn't need fragment)
            progress_pct = (annotated_count / total_sentences) if total_sentences else 0
            st.progress(progress_pct, text=f"Progress: {annotated_count}/{total_sentences} sentences ({progress_pct*100:.1f}%)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", total_sentences)
            with col2:
                st.metric("Annotated", annotated_count)
            with col3:
                st.metric("Remaining", total_sentences - annotated_count)

            st.divider()

            # Main annotation area wrapped in a fragment for instant updates
            @st.fragment
            def render_annotation_section():
                current_idx = st.session_state.current_annotation_idx
                current_row = anno_df.iloc[current_idx]
                
                # Local preset save function within fragment scope
                def local_preset_save(rating_value: int):
                    """Quick save for preset buttons within fragment"""
                    notes = st.session_state.get(f'notes_{current_idx}', '')
                    if rating_value == 0:
                        notes = "Needs review"
                    save_annotation(current_idx, rating_value, notes, move_next=True, total_sentences=total_sentences)
                
                # Navigation callbacks (no rerun needed)
                def nav_first():
                    st.session_state.current_annotation_idx = 0
                    st.session_state.temp_rating.pop(current_idx, None)
                
                def nav_prev():
                    st.session_state.current_annotation_idx = max(0, st.session_state.current_annotation_idx - 1)
                    st.session_state.temp_rating.pop(current_idx, None)
                
                def nav_next():
                    st.session_state.current_annotation_idx = min(total_sentences - 1, st.session_state.current_annotation_idx + 1)
                    st.session_state.temp_rating.pop(current_idx, None)
                
                def nav_last():
                    st.session_state.current_annotation_idx = total_sentences - 1
                    st.session_state.temp_rating.pop(current_idx, None)
                
                def nav_unannotated():
                    curr = st.session_state.current_annotation_idx
                    found = False
                    for i in range(curr + 1, total_sentences):
                        if i not in st.session_state.sentence_annotations:
                            st.session_state.current_annotation_idx = i
                            found = True
                            break
                    if not found:
                        for i in range(0, curr):
                            if i not in st.session_state.sentence_annotations:
                                st.session_state.current_annotation_idx = i
                                found = True
                                break
                    st.session_state.temp_rating.pop(curr, None)

                # Navigation buttons
                nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1.5, 1.5, 1.5, 1.5, 2])
                
                with nav_col1:
                    st.button("⏮️ First", use_container_width=True, disabled=(current_idx == 0), on_click=nav_first, key=f"btn_first_{current_idx}")
                
                with nav_col2:
                    st.button("◀️ Previous", use_container_width=True, disabled=(current_idx == 0), on_click=nav_prev, key=f"btn_prev_{current_idx}")
                
                with nav_col3:
                    st.button("▶️ Next", use_container_width=True, disabled=(current_idx >= total_sentences - 1), on_click=nav_next, key=f"btn_next_{current_idx}")
                
                with nav_col4:
                    st.button("⏭️ Last", use_container_width=True, disabled=(current_idx >= total_sentences - 1), on_click=nav_last, key=f"btn_last_{current_idx}")
                
                with nav_col5:
                    st.button("🎯 Next Unannotated", use_container_width=True, on_click=nav_unannotated, key=f"btn_unannotated_{current_idx}")

                st.divider()

                # Display current sentence
                status_emoji = "✅" if current_idx in st.session_state.sentence_annotations else "⭕"
                st.subheader(f"{status_emoji} Sentence {current_idx + 1} of {total_sentences}")
                st.markdown(f"""
                <div style="padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="margin: 0; color: white; font-size: 28px; line-height: 1.6;">
                        {html.escape(current_row['sentence'])}
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                info_cols = st.columns(3)
                with info_cols[0]:
                    if current_row.get('translation') and pd.notna(current_row['translation']):
                        st.info(f"🌐 **Translation**\n\n{current_row['translation']}")
                with info_cols[1]:
                    if current_row.get('transliteration') and pd.notna(current_row['transliteration']):
                        st.info(f"🔤 **Transliteration**\n\n{current_row['transliteration']}")
                with info_cols[2]:
                    st.info(f"""📊 **Computed Metrics**
                    
- **RH Avg:** {current_row['rh_avg']:.1f}
- **Length:** {int(current_row['sl'])} words
- **Time:** {format_timestamp(current_row.get('start_time'))} → {format_timestamp(current_row.get('end_time'))}
                    """)

                # Audio player (always loaded, no button needed)
                with st.expander("🔊 Listen to Audio", expanded=False):
                    render_audio_player(current_row)

                st.divider()

                existing_annotation = st.session_state.sentence_annotations.get(current_idx, {})
                default_rating = existing_annotation.get('rating', 5)
                default_notes = existing_annotation.get('notes', '')
                current_rating = st.session_state.temp_rating.get(current_idx, default_rating)
                
                st.markdown("**⚡ Quick Rate & Next:**")
                preset_cols = st.columns(4)
                with preset_cols[0]:
                    if st.button("🟢 Easy (3)", use_container_width=True, key=f"preset_easy_{current_idx}"):
                        local_preset_save(3)
                        st.rerun(scope="fragment")
                with preset_cols[1]:
                    if st.button("🟡 Medium (5)", use_container_width=True, key=f"preset_medium_{current_idx}"):
                        local_preset_save(5)
                        st.rerun(scope="fragment")
                with preset_cols[2]:
                    if st.button("🟠 Hard (8)", use_container_width=True, key=f"preset_hard_{current_idx}"):
                        local_preset_save(8)
                        st.rerun(scope="fragment")
                with preset_cols[3]:
                    if st.button("❓ Review (0)", use_container_width=True, key=f"preset_review_{current_idx}"):
                        local_preset_save(0)
                        st.rerun(scope="fragment")

                st.divider()

                with st.expander("🎚️ Fine-tune Rating (Optional)", expanded=False):
                    rating_value = st.number_input(
                        "Detailed Rating (1-10)",
                        min_value=1,
                        max_value=10,
                        value=int(current_rating),
                        step=1,
                        help="Enter a difficulty rating from 1 (easiest) to 10 (hardest)",
                        key=f'rating_{current_idx}'
                    )
                    
                    # Update temp rating directly without callback
                    st.session_state.temp_rating[current_idx] = rating_value

                    if rating_value <= 3:
                        st.success(f"🟢 Easy (Rating: {rating_value}/10)")
                    elif rating_value <= 6:
                        st.warning(f"🟡 Medium (Rating: {rating_value}/10)")
                    elif rating_value <= 8:
                        st.error(f"🟠 Hard (Rating: {rating_value}/10)")
                    else:
                        st.error(f"🔴 Very Hard (Rating: {rating_value}/10)")

                    notes = st.text_area(
                        "Notes (optional)",
                        value=default_notes,
                        placeholder="Add context, vocabulary issues, pronunciation notes...",
                        height=100,
                        key=f'notes_{current_idx}'
                    )

                    detail_col1, detail_col2, detail_col3 = st.columns([2, 1, 1])
                    with detail_col1:
                        if st.button("💾 Save & Next", type="primary", use_container_width=True, key=f"save_next_{current_idx}"):
                            save_annotation(current_idx, rating_value, notes, move_next=True, total_sentences=total_sentences)
                            st.rerun(scope="fragment")
                    with detail_col2:
                        if st.button("💾 Save", use_container_width=True, key=f"save_{current_idx}"):
                            save_annotation(current_idx, rating_value, notes, move_next=False, total_sentences=total_sentences)
                            st.rerun(scope="fragment")
                    with detail_col3:
                        if st.button("🗑️ Clear", use_container_width=True, key=f"clear_{current_idx}"):
                            st.session_state.sentence_annotations.pop(current_idx, None)
                            st.session_state.temp_rating.pop(current_idx, None)
                            st.rerun(scope="fragment")

                if current_idx in st.session_state.sentence_annotations:
                    anno = st.session_state.sentence_annotations[current_idx]
                    timestamp = anno.get('timestamp', 'Unknown')
                    st.caption(f"✏️ Last saved: Rating {anno.get('rating')}/10 at {timestamp}")
            
            # Render the fragment (this updates instantly without full page rerun)
            render_annotation_section()

            # Export section (outside fragment to avoid unnecessary reruns)
            if annotated_count > 0:
                st.divider()
                st.subheader("💾 Export Annotations")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Annotated", annotated_count)
                with col2:
                    ratings = [anno.get('rating', 0) for anno in st.session_state.sentence_annotations.values() if anno.get('rating', 0) > 0]
                    avg_rating = sum(ratings) / len(ratings) if ratings else 0
                    st.metric("Avg Rating", f"{avg_rating:.1f}")
                with col3:
                    easy = sum(1 for anno in st.session_state.sentence_annotations.values() if 1 <= anno.get('rating', 0) <= 3)
                    medium = sum(1 for anno in st.session_state.sentence_annotations.values() if 4 <= anno.get('rating', 0) <= 6)
                    hard = sum(1 for anno in st.session_state.sentence_annotations.values() if 7 <= anno.get('rating', 0) <= 10)
                    st.metric("Easy/Med/Hard", f"{easy}/{medium}/{hard}")

                def prepare_export():
                    st.session_state.export_ready = True

                st.button("📥 Prepare CSV Export", use_container_width=True, type="primary", on_click=prepare_export)

                if st.session_state.get('export_ready', False):
                    with st.spinner("Preparing export..."):
                        export_df = anno_df.copy()
                        ratings_list = []
                        notes_list = []
                        timestamps_list = []
                        for i in range(len(export_df)):
                            anno = st.session_state.sentence_annotations.get(i, {})
                            ratings_list.append(anno.get('rating', ''))
                            notes_list.append(anno.get('notes', ''))
                            timestamps_list.append(str(anno.get('timestamp', '')) if 'timestamp' in anno else '')
                        export_df['difficulty_rating'] = ratings_list
                        export_df['annotation_notes'] = notes_list
                        export_df['annotation_timestamp'] = timestamps_list
                        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download Annotated CSV",
                            data=csv_bytes,
                            file_name=f"annotated_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.session_state.export_ready = False

                if st.button("📊 Show/Hide Statistics", use_container_width=True):
                    st.session_state.show_stats = not st.session_state.get('show_stats', False)

                if st.session_state.get('show_stats', False):
                    st.markdown("#### 📊 Rating Distribution")
                    rating_counts = {}
                    for anno in st.session_state.sentence_annotations.values():
                        rating = anno.get('rating', 0)
                        if rating > 0:
                            rating_counts[rating] = rating_counts.get(rating, 0) + 1
                    if rating_counts:
                        chart_data = pd.DataFrame({
                            'Rating': list(rating_counts.keys()),
                            'Count': list(rating_counts.values())
                        }).sort_values('Rating')
                        st.bar_chart(chart_data.set_index('Rating'))
                    with_notes = sum(1 for anno in st.session_state.sentence_annotations.values() if anno.get('notes'))
                    st.write(f"**Sentences with notes:** {with_notes}/{annotated_count}")
            else:
                st.info("💡 Start annotating to enable export")

            with st.expander("⚡ Speed Tips", expanded=False):
                st.markdown("""
                **Fastest Workflow:**
                1. Read the sentence
                2. Click a **preset button** (Easy/Medium/Hard) - instantly saves and moves to next
                3. Repeat!
                
                **For Fine-tuning:**
                - Open "Fine-tune Rating" expander
                - Enter a number from 1-10
                - Add notes if needed
                - Click "Save & Next"
                
                **Keyboard-Free Annotation:**
                - Use only preset buttons for ultra-fast annotation
                - Avg time per sentence: ~3-5 seconds
                
                **Rating Guide:**
                - **1-3 (Easy)**: Simple vocab, short, clear
                - **4-6 (Medium)**: Moderate complexity
                - **7-8 (Hard)**: Complex grammar/vocab
                - **9-10 (Very Hard)**: Idioms, technical, cultural
                - **0 (Review)**: Unclear audio, errors
                """)
        
with tab6:
    st.header("About This Tool")
    
    st.markdown("""
    ### 🎯 Purpose
    This tool analyzes audio recordings to help language learners identify sentences based on difficulty level.
    
    ### 📊 Metrics Used
    
    - **RH1 & RH2**: Indic Readability scores that measure text complexity
    - **Sentence Length (SL)**: Number of words in each sentence
    - **Combined Score**: Average of readability metrics, sorted with sentence length
    
    ### 🔄 Processing Pipeline
    
    1. **Segmentation**: Audio is split into dialogue chunks using Voice Activity Detection (VAD)
    2. **Transcription**: Each chunk is transcribed using Indic multilingual models
    3. **Alignment**: Sentences are aligned to their timestamps in the original audio
    4. **Analysis**: Difficulty metrics are computed for each sentence
    5. **Visualization**: Results are displayed with audio playback and sorting options
    
    ### 🎚️ VAD Settings Guide
    
    - **Aggressiveness (0-3)**: Higher values filter more aggressively. Use 3 for noisy audio.
    - **Min Speech Length**: Minimum duration (ms) to consider as speech. Lower = detect shorter utterances.
    - **Min Silence**: Silence duration (ms) to mark end of speech segment. Higher = fewer splits.
    
    ### 💡 Tips
    
    - Use **high-quality audio** for best transcription results
    - Adjust **VAD settings** if segmentation misses or over-segments speech
    - **Filter results** to focus on specific difficulty ranges
    - **Sort by difficulty** to create progressive learning sequences
    
    ### 🛠️ Technologies
    
    - Streamlit for UI
    - WebRTC VAD for voice detection
    - Indic Conformer for transcription
    - Custom readability metrics for Indic languages
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Made with ❤️ for language learners | Powered by Streamlit
</div>
""", unsafe_allow_html=True)