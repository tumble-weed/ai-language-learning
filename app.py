"""
Streamlit frontend for Language Learning Difficulty Analyzer
Provides audio upload, processing pipeline, and interactive results visualization.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import tempfile
from typing import List, Dict, Optional
import os
import math
from textwrap import dedent
import dotenv
from rclone_python import rclone
import json
import numpy as np
from datetime import datetime, timezone
from skelo.model.glicko2 import Glicko2Model
from fsrs import Scheduler, Card, Rating
from utils.eval_answer import evaluate_translation
from utils.rclone_helper import upload_files

dotenv.load_dotenv()

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
if 'dropbox_csv_files' not in st.session_state:
    try:
        files = rclone.ls(f"dropbox:{os.getenv('DROPBOX_CSV_FOLDER_PATH','')}")
        st.session_state.dropbox_csv_files = {f["Name"]: f["ID"] for f in files}
    except Exception as e:
        st.error(f"Error listing folder: {e}")
        st.session_state.dropbox_csv_files = {}

# Translation game session state
if 'user_data' not in st.session_state:
    # Try to load user data from Dropbox first, then local file
    try:
        json_folder = os.getenv('DROPBOX_USERS_FOLDER_PATH', '')
        user_file_name = "user_rating.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Download from Dropbox using rclone
                remote_path = f"dropbox:{json_folder}{user_file_name}"
                rclone.copy(remote_path, tmp_dir)
                
                # Read the downloaded file
                downloaded_file = Path(tmp_dir) / user_file_name
                with open(downloaded_file, 'r') as f:
                    raw = json.load(f)
                history = {}
                for sid, obj in raw.get("history", {}).items():
                    card = Card.from_dict(obj["card"])
                    due = datetime.fromisoformat(obj["due"])
                    history[int(sid)] = {"card": card, "due": due}
                raw["history"] = history
                st.session_state.user_data = raw
            except Exception as e:
                pass
    except Exception as e:
        pass
        
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = None
if 'user_answer' not in st.session_state:
    st.session_state.user_answer = ""
if 'show_correct_answer' not in st.session_state:
    st.session_state.show_correct_answer = False
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'game_model' not in st.session_state:
    st.session_state.game_model = Glicko2Model()
if 'game_scheduler' not in st.session_state:
    st.session_state.game_scheduler = Scheduler()

# Translation game constants
MIN_RATING = 600
MAX_RATING = 1800
USER_FILE = "user_rating.json"

# Initialize global master audio element if audio link is available
# if st.session_state.audio_file_link:
#     components.html(f"""
#         <script>
#             (function() {{
#                 // Initialize or update the global master audio
#                 window.top.globalMasterAudio.src = "{st.session_state.audio_file_link}";

#                 // Log for debugging
#                 console.log('Global master audio updated');
#             }})();
#         </script>
#     """, height=0)

if 'audio_file_link' not in st.session_state:
    st.session_state.audio_file_link = None

components.html(f"""
        <script>
            (function() {{
                // Create the global master audio element

                if (window.top.globalMasterAudio) {{
                    window.top.globalMasterAudio.src = "{st.session_state.audio_file_link}";
                    console.log('Global master audio updated:', window.top.globalMasterAudio);
                    return;
                }}
                const audio = document.createElement('audio');
                audio.id = 'globalMasterAudio';
                audio.preload = 'auto';
                audio.style.display = 'none';
                audio.src = "{st.session_state.audio_file_link}";
                window.top.globalMasterAudio = audio;

                // Log for debugging
                console.log('Global master audio created:', audio);
            }})();
        </script>
    """, height=0)


def ensure_results_df() -> Optional[pd.DataFrame]:
    """Return the shared results DataFrame, creating it once if needed."""
    if st.session_state.results_df is not None:
        return st.session_state.results_df
    processed = st.session_state.get('processed_results')
    if processed:
        st.session_state.results_df = pd.DataFrame(processed)
    return st.session_state.results_df


# Translation game helper functions
def rating_to_10_scale(rating):
    """Convert Glicko rating to 1-10 scale."""
    scaled = 1 + 9 * (rating - MIN_RATING) / (MAX_RATING - MIN_RATING)
    return round(min(10, max(1, scaled)), 2)


def scale_10_to_rating(score_10):
    """Convert 1-10 scale to Glicko rating."""
    return MIN_RATING + (score_10 - 1) * (MAX_RATING - MIN_RATING) / 9

def save_user(user):
    """Creating temporary json file and uploading to dropbox using rclone."""
    folder_path = os.getenv('DROPBOX_USERS_FOLDER_PATH', '')
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / USER_FILE
        # Convert Card objects to dicts for JSON serialization
        history_serializable = {}
        for sid, obj in user["history"].items():
            history_serializable[sid] = {
                "card": obj["card"].to_dict(),
                "due": obj["due"].isoformat()
            }
        user_serializable = user.copy()
        user_serializable["history"] = history_serializable

        with open(temp_path, 'w') as f:
            json.dump(user_serializable, f, indent=4)
        
        upload_files([temp_path], dropbox_folder=folder_path)
    



def select_sentence(df, user, user_elo_10, skill_window=1):
    """Select next sentence based on FSRS and difficulty."""
    now = datetime.now(timezone.utc)

    # 1) FSRS due + skill filter
    due_list = []
    for sid, obj in user["history"].items():
        if obj["due"] <= now:
            row = df[df["id"] == sid]
            if row.empty:
                continue
            diff = float(row.iloc[0]["difficulty"])
            if abs(diff - user_elo_10) <= skill_window:
                due_list.append(row.iloc[0])
    
    if due_list:
        return pd.DataFrame(due_list).sample(1).iloc[0]

    # 2) New sampling by difficulty
    center = int(round(user_elo_10))
    min_d = max(1, center - skill_window)
    max_d = min(10, center + skill_window)
    pool = df[(df["difficulty"] >= min_d) & (df["difficulty"] <= max_d)]
    if pool.empty:
        pool = df
    weights = 1 / (1 + np.abs(pool["difficulty"] - user_elo_10))
    return pool.sample(1, weights=weights).iloc[0]


def update_user_rating(user, sentence_difficulty, correct_flag):
    """Update user rating using Glicko2."""
    opp_rating = scale_10_to_rating(sentence_difficulty)
    r1 = (user["rating"], user["rd"], user["sigma"])
    r2 = (opp_rating, 200.0, 0.06)
    
    new_r, new_rd, new_sig = st.session_state.game_model.evolve_rating(
        r1, r2, 1 if correct_flag else 0
    )
    
    user["rating"] = round(new_r, 2)
    user["rd"] = round(new_rd, 2)
    user["sigma"] = round(new_sig, 5)
    user["rating_10"] = rating_to_10_scale(user["rating"])
    return user


def update_fsrs_card(user, sentence_id, correct_flag):
    """Update FSRS card for the sentence."""
    if sentence_id not in user["history"]:
        user["history"][sentence_id] = {
            "card": Card(),
            "due": datetime.now(timezone.utc)
        }
    
    card = user["history"][sentence_id]["card"]
    rating = Rating.Good if correct_flag else Rating.Again
    card, review_log = st.session_state.game_scheduler.review_card(card, rating)
    user["history"][sentence_id]["card"] = card
    user["history"][sentence_id]["due"] = card.due
    return user

# @st.cache_data(show_spinner=False)
# def apply_filters_and_sort(df: pd.DataFrame, sort_by: str, sl_range: tuple, rh_range: tuple) -> pd.DataFrame:
#     """Cache filtered and sorted results to avoid recomputation."""
#     # Apply sorting
#     if sort_by == 'Difficulty (Easy → Hard)':
#         df = df.sort_values(['sl', 'rh_avg'], ascending=[True, True])
#     elif sort_by == 'Difficulty (Hard → Easy)':
#         df = df.sort_values(['sl', 'rh_avg'], ascending=[False, False])
#     elif sort_by == 'Sentence Length':
#         df = df.sort_values('sl', ascending=True)
#     else:  # Audio File
#         df = df.sort_values('audio_file', ascending=True)
    
#     # Reset index
#     df = df.reset_index(drop=True)
    
#     # Apply filters
#     filtered_df = df[
#         (df['sl'] >= sl_range[0]) & 
#         (df['sl'] <= sl_range[1]) &
#         (df['rh_avg'] >= rh_range[0]) & 
#         (df['rh_avg'] <= rh_range[1])
#     ]
    
#     return filtered_df

# @st.cache_data(show_spinner=False)
# def extract_audio_clip(audio_data, samplerate: int, start: Optional[float], end: Optional[float]) -> Optional[bytes]:
#     """Return WAV bytes for the requested time window from already loaded audio data."""
#     if audio_data is None or start is None or end is None:
#         return None
    
#     try:
#         start_val = float(start)
#         end_val = float(end)
#     except (TypeError, ValueError):
#         return None
    
#     if end_val <= start_val:
#         return None

#     start_sample = max(0, int(start_val * samplerate))
#     end_sample = min(len(audio_data), int(end_val * samplerate))
#     if start_sample >= end_sample:
#         return None

#     clip = audio_data[start_sample:end_sample]
#     buffer = io.BytesIO()
#     sf.write(buffer, clip, samplerate, format='WAV')
#     return buffer.getvalue()


# def render_audio_player(row: Dict) -> None:
#     """Display audio snippet using original audio file if available."""

#     # st.audio(st.session_state.audio_file_path, format='audio/wav', start=row.get('start'), end=row.get('end'))
#     # return

#     start = row["start"]
#     end = row["end"]

#     safe_id = f"a_{str(start).replace('.', '_')}_{str(end).replace('.', '_')}"

#     audio_html = f"""
#     <audio id="{safe_id}" preload="none" src="{st.session_state.audio_file_link}#t={start},{end}" onplay="window.snippetPlayer(this, {start}, {end})" controls ></audio>
#     <script>
#     // 1. Function Guard: Define the global snippetPlayer function only once
#     if (!window.snippetPlayer) {{
#         window.snippetPlayer = function(audio, start, end) {{
#             // console.log("Snippet player invoked:", audio, start, end);
#             if (!audio) return;

#             // Clear any previously running interval for this audio element
#             if (audio._snippetInterval) {{
#                 clearInterval(audio._snippetInterval);
#             }}

#             // Snap to the start time (necessary if user has scrubbed or it's the first play)
#             if (audio.currentTime < start || audio.currentTime >= end) {{
#                 audio.currentTime = start;
#             }}

#             // 2. Loop/Stop Logic: Set a new interval to monitor current time
#             audio._snippetInterval = setInterval(() => {{
#                 if (audio.currentTime >= end) {{
#                     audio.pause();
#                     // Reset to start time for a clean restart if user plays again
#                     audio.currentTime = start;
#                     clearInterval(audio._snippetInterval);
#                 }}
#             }}, 50); // Using 50ms for better precision than 100ms
#         }};
#     }}
#     </script>
#     """

#     components.html(audio_html, height=80)
#     return

#     # Extract clip from cached audio data
#     if st.session_state.original_audio_data is not None:
#         start = row.get('start')
#         end = row.get('end')
        
#         # Create unique cache key based on start and end times
#         cache_key = f"{start}_{end}"
        
#         # Check if snippet is already in cache
#         if cache_key not in st.session_state.audio_snippets_cache:
#             # Extract and cache the snippet
#             snippet = extract_audio_clip(
#                 st.session_state.original_audio_data,
#                 st.session_state.original_audio_samplerate,
#                 start,
#                 end
#             )
#             st.session_state.audio_snippets_cache[cache_key] = snippet
#         else:
#             # Load from cache
#             snippet = st.session_state.audio_snippets_cache[cache_key]
        
#         caption_parts = []
#         if st.session_state.audio_file_path:
#             caption_parts.append(st.session_state.audio_file_path.name)
#         if start is not None and end is not None:
#             try:
#                 caption_parts.append(f"{float(start):.2f}s → {float(end):.2f}s")
#             except (TypeError, ValueError):
#                 pass
#         caption_text = " | ".join(caption_parts) if caption_parts else "Audio clip"
        
#         if snippet:
#             st.audio(snippet, format='audio/wav')
#             st.caption(f"📁 {caption_text}")
#             return
    
#     st.warning("⚠️ Audio clip not available for this sentence")


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
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')

    start_vals = df['start'].dropna()
    end_vals = df['end'].dropna()
    needs_scaling = False
    if not start_vals.empty and start_vals.gt(1000).any():
        needs_scaling = True
    if not end_vals.empty and end_vals.gt(1000).any():
        needs_scaling = True

    if needs_scaling:
        df['start'] = df['start'] / 1000.0
        df['end'] = df['end'] / 1000.0


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

# Title and description
st.title("🎙️ Language Learning Difficulty Analyzer")
st.markdown("""
Analyze audio recordings to identify sentence difficulty based on readability metrics and sentence length.
Upload an audio file, process it through the pipeline, and explore the results interactively.
""")

tab1, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📂 Load Results", "📊 Results", "🎯 Practice", "🎮 Translation Game", "📝 Annotate", "ℹ️ About"])

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
            # column_mapping = {
            #     'Audio_file': 'audio_file',
            #     'Text': 'sentence',
            #     'RH1_Result': 'rh1',
            #     'RH2_Result': 'rh2',
            #     'RH_Average': 'rh_avg',
            #     'SL_Result': 'sl',
            #     'Chunk_ID': 'audio_file',
            #     'WFR_Result': 'wfr',
            #     'Transliteration': 'transliteration',
            #     'Translation': 'translation',
            #     'Original_audio_file': 'original_audio_file',
            #     'start': 'start',
            #     'end': 'end',
            #     'Difficulty': 'difficulty',
            # }
            
            # df = df.rename(columns=column_mapping)
            
            # Required columns check
            # required_cols = ['sentence', 'rh1', 'rh2', 'rh_avg', 'sl']
            # missing_cols = [col for col in required_cols if col not in df.columns]
            
            # if missing_cols:
            #     st.error(f"❌ CSV is missing required columns: {', '.join(missing_cols)}")
            #     st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            #     return False
            
            # # Handle missing/optional columns with defaults
            # if 'audio_file' not in df.columns:
            #     df['audio_file'] = 'N/A'
            #     st.warning("⚠️ CSV doesn't have audio file column. Audio playback will not be available.")
            
            for col, default in [
                ('wfr', None),
                ('transliteration', None),
                ('translation', None),
                ('rh1', 0),
                ('rh2', 0),
                ('rh_avg', 0),
                ('sl', 0)
            ]:
                if col not in df.columns:
                    df[col] = default
            
            normalize_timestamp_units(df)
            
            # Load original audio file if available
            original_audio_path = None
            if 'original_audio_file' in df.columns and df['original_audio_file'].notna().any():
                # Get the first non-null original_audio_file path
                original_audio_path = df.loc[df['original_audio_file'].notna(), 'original_audio_file'].iloc[0]
                
                # Below commented code is used when audio file have local paths (For testing purpose)
                # original_audio_path = Path(original_audio_path)
                # if original_audio_path.exists():
                #     st.session_state.audio_file_path = original_audio_path
                    
                #     # Load the entire audio file into memory
                #     try:
                #         audio_data, samplerate = sf.read(str(original_audio_path))
                #         st.session_state.original_audio_data = audio_data
                #         st.session_state.original_audio_samplerate = samplerate
                #         st.success(f"✅ Loaded original audio file: {original_audio_path.name}")
                #     except Exception as e:
                #         st.warning(f"⚠️ Could not load audio data from {original_audio_path.name}: {str(e)}")
                #         st.session_state.audio_file_path = None
                #         st.session_state.original_audio_data = None
                #         st.session_state.original_audio_samplerate = None
                # else:
                #     st.warning(f"⚠️ Original audio file not found: {original_audio_path}")
                #     st.session_state.audio_file_path = None
                #     st.session_state.original_audio_data = None
                #     st.session_state.original_audio_samplerate = None


                # Below code is to download audio file from google drive URL
                # with tempfile.TemporaryDirectory() as tempdir:

                #     temp_path = Path(tempdir) / "audio.wav"

                #     gdown.download(original_audio_path, str(temp_path), quiet=False, fuzzy=True)

                #     # Load audio directly from the temp file
                #     audio_data, samplerate = sf.read(str(temp_path))
                #     st.session_state.audio_file_path = temp_path
                #     st.session_state.original_audio_data = audio_data
                #     st.session_state.original_audio_samplerate = samplerate
                #     st.success(f"✅ Downloaded and loaded original audio file from URL")

                # Below code is for directly using audio link without downloading
                st.session_state.audio_file_link = original_audio_path
                # st.info(f"Using audio link from CSV: {original_audio_path}")
                if st.session_state.audio_file_link:
                    components.html(f"""
                        <script>
                            (function() {{
                                // Initialize or update the global master audio
                                window.top.globalMasterAudio.src = "{st.session_state.audio_file_link}";

                                // Log for debugging
                                console.log('Global master audio initialized');
                            }})();
                        </script>
                    """, height=0)
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
    # st.subheader("📁 Quick Load from Output Folder")
    
    # output_dir = Path(__file__).resolve().parent / "output"
    
    # if output_dir.exists():
    #     csv_files = list(output_dir.glob("**/*.csv"))
        
    #     if csv_files:
    #         csv_options = {f"{f.parent.name}/{f.name}": f for f in csv_files}
            
    #         selected_file = st.selectbox(
    #             "Select a results file from output folder:",
    #             options=["-- Select a file --"] + list(csv_options.keys())
    #         )
            
    #         if selected_file != "-- Select a file --" and st.button("📥 Load Selected File", type="primary"):
    #             if load_csv_to_results(csv_options[selected_file]):
    #                 st.success(f"✅ Loaded {len(st.session_state.processed_results)} sentences from {csv_options[selected_file].name}!")
    #                 st.info("👉 Go to the **Results** tab to view the data")
    #     else:
    #         st.info("No CSV files found in output folder. Process an audio file first or upload a CSV above.")
    # else:
    #     st.info("Output folder not found. Process an audio file first or upload a CSV above.")

    st.subheader("📁 Quick Load from Drive")

    if st.session_state.dropbox_csv_files:
        selected_drive_file = st.selectbox(
            "Select a results CSV file from Dropbox:",
            options=["-- Select a file --"] + list(st.session_state.dropbox_csv_files.keys())
        )
        
        if selected_drive_file != "-- Select a file --" and st.button("📥 Load Selected File", type="primary"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    # Download from Dropbox using rclone
                    remote_path = f"dropbox:{os.getenv('DROPBOX_CSV_FOLDER_PATH','')}{selected_drive_file}"
                    rclone.copy(remote_path, tmp_dir)
                    
                    # Read the downloaded file
                    downloaded_file = Path(tmp_dir) / selected_drive_file
                    if load_csv_to_results(downloaded_file):
                        st.success(f"✅ Loaded {len(st.session_state.processed_results)} sentences from {selected_drive_file}!")
                        st.info("👉 Go to the **Results** tab to view the data")
                except Exception as e:
                    st.error(f"❌ Error downloading file from Dropbox: {str(e)}")
                    st.exception(e)

with tab3:
    st.header("Analysis Results")
    
    if st.session_state.processing_complete and st.session_state.processed_results:
        # Ensure tabs are pre-computed
        df = ensure_results_df()
        
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
        # st.subheader("🔍 Filter Results")
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     sl_range = st.slider(
        #         "Sentence Length Range",
        #         min_value=int(df['sl'].min()),
        #         max_value=int(df['sl'].max()),
        #         value=(int(df['sl'].min()), int(df['sl'].max())),
        #         key="results_sl_range"
        #     )
        
        # with col2:
        #     rh_range = st.slider(
        #         "RH Average Range",
        #         min_value=float(df['rh_avg'].min()),
        #         max_value=float(df['rh_avg'].max()),
        #         value=(float(df['rh_avg'].min()), float(df['rh_avg'].max())),
        #         step=0.1,
        #         key="results_rh_range"
        #     )
        
        # Use cached filtering and sorting
        # filtered_df = apply_filters_and_sort(df, sort_by, sl_range, rh_range)
        
        # st.info(f"Showing {len(filtered_df)} of {len(df)} sentences")
        
        st.subheader("📋 Sentence Grid")
        
        # Prepare data for JavaScript
        import json
        
        sentences_data = []
        for idx, row in enumerate(df.itertuples()):
            sentence_obj = {
                'idx': idx,
                'sentence': str(row.sentence),
                'translation': str(getattr(row, 'translation', '')) if pd.notna(getattr(row, 'translation', None)) else '',
                'transliteration': str(getattr(row, 'transliteration', '')) if pd.notna(getattr(row, 'transliteration', None)) else '',
                'rh_avg': float(row.rh_avg),
                'rh1': float(row.rh1),
                'rh2': float(row.rh2),
                'sl': int(row.sl),
                'audio_file': str(row.audio_file),
                'audio_link': st.session_state.audio_file_link if st.session_state.audio_file_link else '',
                'start': str(getattr(row, 'start', '')),
                'end': str(getattr(row, 'end', '')),
                'start_time_sec': float(getattr(row, 'start', 0)) if pd.notna(getattr(row, 'start', None)) else 0,
                'end_time_sec': float(getattr(row, 'end', 0)) if pd.notna(getattr(row, 'end', None)) else 0
            }
            sentences_data.append(sentence_obj)
        
        sentences_json = json.dumps(sentences_data)
        
        # Render using components.html()
        results_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 0; margin: 0; background: transparent; }}
                .grid-container {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                .sentence-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .sentence-header {{ font-weight: 600; color: #666; margin-bottom: 10px; font-size: 14px; }}
                .sentence-text {{ font-size: 18px; font-weight: 500; margin: 12px 0; line-height: 1.6; color: #333; }}
                .expander {{ margin: 10px 0; }}
                .expander-btn {{ background: #f0f2f6; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; width: 100%; text-align: left; font-weight: 500; }}
                .expander-btn:hover {{ background: #e5e7eb; }}
                .expander-content {{ display: none; padding: 10px; background: #f9fafb; border-radius: 6px; margin-top: 5px; }}
                .expander-content.show {{ display: block; }}
                .audio-section {{ margin: 15px 0; padding: 15px; background: #f9fafb; border-radius: 8px; }}
                .custom-audio-controls {{ display: flex; align-items: center; gap: 15px; margin-top: 10px; }}
                .play-pause-btn {{ width: 40px; height: 40px; border-radius: 50%; border: none; background: #3b82f6; color: white; font-size: 16px; cursor: pointer; transition: all 0.2s; }}
                .play-pause-btn:hover {{ background: #2563eb; transform: scale(1.05); }}
                .progress-container {{ flex: 1; height: 6px; background: #e5e7eb; border-radius: 3px; cursor: pointer; position: relative; }}
                .progress-bar {{ height: 100%; background: #3b82f6; border-radius: 3px; width: 0%; transition: width 0.1s; }}
                .time-display {{ font-size: 12px; color: #666; min-width: 80px; text-align: right; }}
                .file-caption {{ font-size: 12px; color: #666; margin-top: 5px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px; }}
                .metric-box {{ background: #f9fafb; padding: 10px; border-radius: 6px; text-align: center; }}
                .metric-label {{ font-size: 11px; color: #666; margin-bottom: 4px; }}
                .metric-value {{ font-size: 18px; font-weight: 600; color: #333; }}
                #masterAudio {{ display: none; }}
            </style>
        </head>
        <body>
            <div class="grid-container" id="sentencesGrid"></div>
            
            <script>
                const sentences = {sentences_json};
                // Access global master audio - try multiple levels up the window hierarchy
                const masterAudio = (window.parent.parent && window.parent.parent.globalMasterAudio) || 
                                   (window.parent && window.parent.globalMasterAudio) || 
                                   (window.top && window.top.globalMasterAudio);
                let currentPlayingIdx = -1;
                let snippetInterval = null;
                
                function toggleExpander(idx, type) {{
                    const content = document.getElementById(`expander-${{type}}-${{idx}}`);
                    content.classList.toggle('show');
                }}
                
                function formatTime(seconds) {{
                    const mins = Math.floor(seconds / 60);
                    const secs = Math.floor(seconds % 60);
                    return mins + ':' + (secs < 10 ? '0' : '') + secs;
                }}
                
                function updateProgress(idx, startTime, endTime) {{
                    if (!masterAudio || endTime <= startTime) return;
                    const currentTime = masterAudio.currentTime;
                    const duration = endTime - startTime;
                    const elapsed = Math.max(0, Math.min(currentTime - startTime, duration));
                    const progress = (elapsed / duration) * 100;
                    
                    document.getElementById(`progress-${{idx}}`).style.width = progress + '%';
                    document.getElementById(`time-${{idx}}`).textContent = formatTime(elapsed) + ' / ' + formatTime(duration);
                }}
                
                function togglePlayPause(idx, startTime, endTime) {{
                    if (currentPlayingIdx === idx) {{
                        pauseAudio(idx);
                    }} else {{
                        if (currentPlayingIdx !== -1) pauseAudio(currentPlayingIdx);
                        playAudio(idx, startTime, endTime);
                    }}
                }}
                
                function playAudio(idx, startTime, endTime) {{
                    if (!masterAudio) return;
                    
                    if (masterAudio.currentTime < startTime || masterAudio.currentTime >= endTime) {{
                        masterAudio.currentTime = startTime;
                    }}
                    
                    masterAudio.play();
                    currentPlayingIdx = idx;
                    document.getElementById(`play-btn-${{idx}}`).textContent = '⏸️';
                    
                    if (snippetInterval) clearInterval(snippetInterval);
                    
                    snippetInterval = setInterval(() => {{
                        updateProgress(idx, startTime, endTime);
                        if (masterAudio.currentTime >= endTime) {{
                            pauseAudio(idx);
                            masterAudio.currentTime = startTime;
                            updateProgress(idx, startTime, endTime);
                        }}
                    }}, 50);
                }}
                
                function pauseAudio(idx) {{
                    if (!masterAudio) return;
                    masterAudio.pause();
                    currentPlayingIdx = -1;
                    document.getElementById(`play-btn-${{idx}}`).textContent = '▶️';
                    if (snippetInterval) {{
                        clearInterval(snippetInterval);
                        snippetInterval = null;
                    }}
                }}
                
                function seekAudio(event, idx, startTime, endTime) {{
                    if (!masterAudio || endTime <= startTime) return;
                    const rect = event.currentTarget.getBoundingClientRect();
                    const clickX = event.clientX - rect.left;
                    const percentage = clickX / rect.width;
                    const duration = endTime - startTime;
                    const newTime = startTime + (duration * percentage);
                    masterAudio.currentTime = Math.max(startTime, Math.min(newTime, endTime));
                    updateProgress(idx, startTime, endTime);
                }}
                
                function escapeHtml(text) {{
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }}
                
                // Render all sentences
                const grid = document.getElementById('sentencesGrid');
                sentences.forEach((s, idx) => {{
                    const card = document.createElement('div');
                    card.className = 'sentence-card';
                    
                    let html = `
                        <div class="sentence-header">Sentence ${{idx + 1}}</div>
                        <div class="sentence-text">${{escapeHtml(s.sentence)}}</div>
                    `;
                    
                    if (s.transliteration) {{
                        html += `
                            <div class="expander">
                                <button class="expander-btn" onclick="toggleExpander(${{idx}}, 'translit')">🔤 Show Transliteration</button>
                                <div class="expander-content" id="expander-translit-${{idx}}">${{escapeHtml(s.transliteration)}}</div>
                            </div>
                        `;
                    }}
                    
                    if (s.translation) {{
                        html += `
                            <div class="expander">
                                <button class="expander-btn" onclick="toggleExpander(${{idx}}, 'trans')">🌐 Show Translation</button>
                                <div class="expander-content" id="expander-trans-${{idx}}">${{escapeHtml(s.translation)}}</div>
                            </div>
                        `;
                    }}
                    
                    if (s.audio_link && masterAudio) {{
                        const duration = s.end_time_sec - s.start_time_sec;
                        const formattedDuration = formatTime(duration);
                        html += `
                            <div class="audio-section">
                                <div class="custom-audio-controls">
                                    <button class="play-pause-btn" id="play-btn-${{idx}}" onclick="togglePlayPause(${{idx}}, ${{s.start_time_sec}}, ${{s.end_time_sec}})">▶️</button>
                                    <div class="progress-container" onclick="seekAudio(event, ${{idx}}, ${{s.start_time_sec}}, ${{s.end_time_sec}})">
                                        <div class="progress-bar" id="progress-${{idx}}"></div>
                                    </div>
                                    <div class="time-display" id="time-${{idx}}">0:00 / ${{formattedDuration}}</div>
                                </div>
                                <div class="file-caption">📁 ${{s.audio_file.split('/').pop().split('\\\\').pop()}}</div>
                            </div>
                        `;
                    }}
                    
                    html += `
                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-label">Length</div>
                                <div class="metric-value">${{s.sl}}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">RH Avg</div>
                                <div class="metric-value">${{s.rh_avg.toFixed(2)}}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">RH1</div>
                                <div class="metric-value">${{s.rh1.toFixed(2)}}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">RH2</div>
                                <div class="metric-value">${{s.rh2.toFixed(2)}}</div>
                            </div>
                        </div>
                    `;
                    
                    card.innerHTML = html;
                    grid.appendChild(card);
                }});
            </script>
        </body>
        </html>
        """
        
        components.html(results_html, height=800, scrolling=True)
        
        # Download results
        st.divider()
        st.subheader("💾 Export Results")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
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
        practice_df = ensure_results_df()
        
        if practice_df is None or len(practice_df) == 0:
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
            
            # st.divider()

            # Enhanced Controls with better UX
            # st.markdown("### 🎛️ Practice Controls")
            
            # controls_row1_col1, controls_row1_col2 = st.columns([3, 1])
            
            # with controls_row1_col1:
            #     search_query = st.text_input(
            #         "🔍 Search translation or sentence",
            #         placeholder="Type keywords to filter practice cards...",
            #         key="practice_search",
            #         help="Search in both translations and original sentences"
            #     )
            
            # with controls_row1_col2:
            #     shuffle_cards = st.checkbox("🔀 Shuffle", value=False, key="practice_shuffle", help="Randomize card order")
            
            # controls_row2_col1, controls_row2_col2 = st.columns([2, 2])
            
            # with controls_row2_col1:
            #     selected_difficulties = st.multiselect(
            #         "🎚️ Difficulty Levels",
            #         options=difficulty_options,
            #         default=difficulty_options,
            #         key="practice_difficulty",
            #         help="Filter by difficulty level"
            #     )
            
            # with controls_row2_col2:
            #     sl_min = int(practice_df['sl'].min())
            #     sl_max = int(practice_df['sl'].max())
            #     sentence_length_filter = st.slider(
            #         "📏 Sentence Length Range",
            #         min_value=sl_min,
            #         max_value=sl_max,
            #         value=(sl_min, sl_max),
            #         key="practice_sl_filter",
            #         help="Filter by number of words"
            #     )

            # Apply filters
            # filtered_df = practice_df
            # active_difficulties = selected_difficulties if selected_difficulties else difficulty_options
            # filtered_df = filtered_df[filtered_df['difficulty_label'].isin(active_difficulties)]
            
            # # Apply sentence length filter
            # filtered_df = filtered_df[
            #     (filtered_df['sl'] >= sentence_length_filter[0]) & 
            #     (filtered_df['sl'] <= sentence_length_filter[1])
            # ]

            # if search_query:
            #     mask = (
            #         filtered_df['translation'].str.contains(search_query, case=False, na=False) |
            #         filtered_df['sentence'].str.contains(search_query, case=False, na=False)
            #     )
            #     filtered_df = filtered_df[mask]

            if practice_df.empty:
                st.warning("⚠️ No sentences match your current filters. Try widening the search or selecting more difficulty levels.")
            else:
                # if shuffle_cards:
                #     filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
                
                practice_df = practice_df.reset_index(drop=True)
                
                st.divider()
                
                # List view layout
                st.subheader(f"📚 Practice Cards ({len(practice_df)} sentences)")
                
                # Prepare data for JavaScript
                import json
                
                practice_sentences = []
                for idx, row in enumerate(practice_df.itertuples()):
                    difficulty_label = row.difficulty_label
                    difficulty_emoji = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(difficulty_label, "")
                    
                    sentence_obj = {
                        'idx': idx,
                        'sentence': str(row.sentence),
                        'translation': str(row.translation),
                        'transliteration': str(getattr(row, 'transliteration', '')) if pd.notna(getattr(row, 'transliteration', None)) else '',
                        'rh_avg': float(row.rh_avg),
                        'rh1': float(row.rh1),
                        'rh2': float(row.rh2),
                        'sl': int(row.sl),
                        'audio_file': str(row.audio_file),
                        'audio_link': st.session_state.audio_file_link if st.session_state.audio_file_link else '',
                        'start': str(getattr(row, 'start', '')),
                        'end': str(getattr(row, 'end', '')),
                        'start_time_sec': float(getattr(row, 'start', 0)) if pd.notna(getattr(row, 'start', None)) else 0,
                        'end_time_sec': float(getattr(row, 'end', 0)) if pd.notna(getattr(row, 'end', None)) else 0,
                        'difficulty_label': difficulty_label,
                        'difficulty_emoji': difficulty_emoji
                    }
                    practice_sentences.append(sentence_obj)
                
                sentences_json = json.dumps(practice_sentences)
                
                # Render using components.html()
                practice_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 0; margin: 0; background: transparent; }}
                        .practice-card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                        .card-header {{ font-size: 18px; font-weight: 600; margin-bottom: 15px; color: #333; }}
                        .translation-box {{ background: #e0f2fe; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #0284c7; }}
                        .translation-text {{ font-size: 16px; line-height: 1.6; color: #0c4a6e; font-weight: 500; }}
                        .hint {{ font-size: 13px; color: #666; margin: 10px 0; font-style: italic; }}
                        .metrics-compact {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 15px 0; }}
                        .metric-box {{ background: #f9fafb; padding: 12px; border-radius: 6px; }}
                        .metric-label {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
                        .metric-value {{ font-size: 16px; font-weight: 600; color: #333; }}
                        .timestamp {{ font-size: 12px; color: #666; margin-top: 5px; }}
                        .reveal-section {{ margin-top: 15px; }}
                        .reveal-btn {{ background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; width: 100%; }}
                        .reveal-btn:hover {{ background: #2563eb; }}
                        .reveal-content {{ display: none; margin-top: 15px; padding: 15px; background: #f9fafb; border-radius: 8px; }}
                        .reveal-content.show {{ display: block; }}
                        .answer-text {{ font-size: 16px; font-weight: 500; margin: 10px 0; line-height: 1.6; }}
                        .section-label {{ font-weight: 600; margin: 10px 0 5px 0; color: #333; }}
                        .audio-section {{ margin: 15px 0; padding: 15px; background: white; border-radius: 8px; border: 1px solid #e5e7eb; }}
                        .custom-audio-controls {{ display: flex; align-items: center; gap: 15px; margin-top: 10px; }}
                        .play-pause-btn {{ width: 40px; height: 40px; border-radius: 50%; border: none; background: #3b82f6; color: white; font-size: 16px; cursor: pointer; transition: all 0.2s; }}
                        .play-pause-btn:hover {{ background: #2563eb; transform: scale(1.05); }}
                        .progress-container {{ flex: 1; height: 6px; background: #e5e7eb; border-radius: 3px; cursor: pointer; position: relative; }}
                        .progress-bar {{ height: 100%; background: #3b82f6; border-radius: 3px; width: 0%; transition: width 0.1s; }}
                        .time-display {{ font-size: 12px; color: #666; min-width: 80px; text-align: right; }}
                        .file-caption {{ font-size: 12px; color: #666; margin-top: 5px; }}
                        #masterAudio {{ display: none; }}
                    </style>
                </head>
                <body>
                    <div id="practiceContainer"></div>
                    
                    <script>
                        const sentences = {sentences_json};
                        // Access global master audio - try multiple levels up the window hierarchy
                        const masterAudio = (window.parent.parent && window.parent.parent.globalMasterAudio) || 
                                           (window.parent && window.parent.globalMasterAudio) || 
                                           (window.top && window.top.globalMasterAudio);
                        let currentPlayingIdx = -1;
                        let snippetInterval = null;
                        
                        function formatTime(seconds) {{
                            const mins = Math.floor(seconds / 60);
                            const secs = Math.floor(seconds % 60);
                            return mins + ':' + (secs < 10 ? '0' : '') + secs;
                        }}
                        
                        function formatTimestamp(seconds) {{
                            if (!seconds || seconds === 'nan' || seconds === '') return '--';
                            const total = parseFloat(seconds);
                            if (isNaN(total) || total < 0) return '--';
                            const mins = Math.floor(total / 60);
                            const secs = (total - mins * 60).toFixed(2);
                            return String(mins).padStart(2, '0') + ':' + String(secs).padStart(5, '0');
                        }}
                        
                        function toggleReveal(idx) {{
                            const content = document.getElementById(`reveal-${{idx}}`);
                            content.classList.toggle('show');
                        }}
                        
                        function updateProgress(idx, startTime, endTime) {{
                            if (!masterAudio || endTime <= startTime) return;
                            const currentTime = masterAudio.currentTime;
                            const duration = endTime - startTime;
                            const elapsed = Math.max(0, Math.min(currentTime - startTime, duration));
                            const progress = (elapsed / duration) * 100;
                            
                            const progressBar = document.getElementById(`progress-${{idx}}`);
                            const timeDisplay = document.getElementById(`time-${{idx}}`);
                            if (progressBar) progressBar.style.width = progress + '%';
                            if (timeDisplay) timeDisplay.textContent = formatTime(elapsed) + ' / ' + formatTime(duration);
                        }}
                        
                        function togglePlayPause(idx, startTime, endTime) {{
                            if (currentPlayingIdx === idx) {{
                                pauseAudio(idx);
                            }} else {{
                                if (currentPlayingIdx !== -1) pauseAudio(currentPlayingIdx);
                                playAudio(idx, startTime, endTime);
                            }}
                        }}
                        
                        function playAudio(idx, startTime, endTime) {{
                            if (!masterAudio) return;
                            
                            if (masterAudio.currentTime < startTime || masterAudio.currentTime >= endTime) {{
                                masterAudio.currentTime = startTime;
                            }}
                            
                            masterAudio.play();
                            currentPlayingIdx = idx;
                            const btn = document.getElementById(`play-btn-${{idx}}`);
                            if (btn) btn.textContent = '⏸️';
                            
                            if (snippetInterval) clearInterval(snippetInterval);
                            
                            snippetInterval = setInterval(() => {{
                                updateProgress(idx, startTime, endTime);
                                if (masterAudio.currentTime >= endTime) {{
                                    pauseAudio(idx);
                                    masterAudio.currentTime = startTime;
                                    updateProgress(idx, startTime, endTime);
                                }}
                            }}, 50);
                        }}
                        
                        function pauseAudio(idx) {{
                            if (!masterAudio) return;
                            masterAudio.pause();
                            currentPlayingIdx = -1;
                            const btn = document.getElementById(`play-btn-${{idx}}`);
                            if (btn) btn.textContent = '▶️';
                            if (snippetInterval) {{
                                clearInterval(snippetInterval);
                                snippetInterval = null;
                            }}
                        }}
                        
                        function seekAudio(event, idx, startTime, endTime) {{
                            if (!masterAudio || endTime <= startTime) return;
                            const rect = event.currentTarget.getBoundingClientRect();
                            const clickX = event.clientX - rect.left;
                            const percentage = clickX / rect.width;
                            const duration = endTime - startTime;
                            const newTime = startTime + (duration * percentage);
                            masterAudio.currentTime = Math.max(startTime, Math.min(newTime, endTime));
                            updateProgress(idx, startTime, endTime);
                        }}
                        
                        function escapeHtml(text) {{
                            const div = document.createElement('div');
                            div.textContent = text;
                            return div.innerHTML;
                        }}
                        
                        // Render all practice cards
                        const container = document.getElementById('practiceContainer');
                        sentences.forEach((s, idx) => {{
                            const card = document.createElement('div');
                            card.className = 'practice-card';
                            
                            let html = `
                                <div class="card-header">${{s.difficulty_emoji}} Sentence #${{idx + 1}} - ${{s.difficulty_label}}</div>
                                <div class="translation-box">
                                    <div style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">🌐 Translation</div>
                                    <div class="translation-text">${{escapeHtml(s.translation)}}</div>
                                </div>
                                <div class="hint">💡 Try to recall the original sentence before revealing!</div>
                                
                                <div class="metrics-compact">
                                    <div class="metric-box">
                                        <div class="metric-label">Length</div>
                                        <div class="metric-value">${{s.sl}} words</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-label">RH Avg</div>
                                        <div class="metric-value">${{s.rh_avg.toFixed(2)}}</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-label">RH1</div>
                                        <div class="metric-value">${{s.rh1.toFixed(2)}}</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-label">RH2</div>
                                        <div class="metric-value">${{s.rh2.toFixed(2)}}</div>
                                    </div>
                                </div>
                            `;
                            
                            const startLabel = formatTimestamp(s.start_time_sec);
                            const endLabel = formatTimestamp(s.end_time_sec);
                            if (startLabel !== '--' || endLabel !== '--') {{
                                html += `<div class="timestamp">⏱️ ${{startLabel}} → ${{endLabel}}</div>`;
                            }}
                            
                            html += `
                                <div class="reveal-section">
                                    <button class="reveal-btn" onclick="toggleReveal(${{idx}})">👁️ Reveal Answer</button>
                                    <div class="reveal-content" id="reveal-${{idx}}">
                                        <div class="section-label">Original Sentence:</div>
                                        <div class="answer-text">${{escapeHtml(s.sentence)}}</div>
                            `;
                            
                            if (s.transliteration) {{
                                html += `
                                    <div class="section-label" style="margin-top: 15px;">Transliteration:</div>
                                    <div class="answer-text">${{escapeHtml(s.transliteration)}}</div>
                                `;
                            }}
                            
                            if (s.audio_link && masterAudio) {{
                                const duration = s.end_time_sec - s.start_time_sec;
                                const formattedDuration = formatTime(duration);
                                html += `
                                    <div class="section-label" style="margin-top: 15px;">Audio:</div>
                                    <div class="audio-section">
                                        <div class="custom-audio-controls">
                                            <button class="play-pause-btn" id="play-btn-${{idx}}" onclick="togglePlayPause(${{idx}}, ${{s.start_time_sec}}, ${{s.end_time_sec}})">▶️</button>
                                            <div class="progress-container" onclick="seekAudio(event, ${{idx}}, ${{s.start_time_sec}}, ${{s.end_time_sec}})">
                                                <div class="progress-bar" id="progress-${{idx}}"></div>
                                            </div>
                                            <div class="time-display" id="time-${{idx}}">0:00 / ${{formattedDuration}}</div>
                                        </div>
                                        <div class="file-caption">📁 ${{s.audio_file.split('/').pop().split('\\\\').pop()}}</div>
                                    </div>
                                `;
                            }}
                            
                            html += `
                                    </div>
                                </div>
                            `;
                            
                            card.innerHTML = html;
                            container.appendChild(card);
                        }});
                    </script>
                </body>
                </html>
                """
                
                components.html(practice_html, height=800, scrolling=True)
    
    else:
        st.info("👈 Load results from the **Load Results** tab to start practicing!")

with tab5:
    st.header("🎮 Translation Game")
    st.markdown("""
    Practice translation with adaptive difficulty! The game adjusts to your skill level using Glicko2 and FSRS algorithms.
    """)

    if not (st.session_state.processing_complete and st.session_state.processed_results):
        st.info("⚠️ Please load results from the 'Load Results' tab or process an audio file first.")
    else:
        df = ensure_results_df()
        
        # Check if required columns exist
        if 'difficulty' not in df.columns:
            st.warning("⚠️ The loaded data doesn't have a 'difficulty' column. Please ensure your CSV includes sentence difficulty ratings.")
        elif 'translation' not in df.columns:
            st.warning("⚠️ The loaded data doesn't have a 'translation' column. Translation game requires both source sentences and translations.")
        else:
            # Get user data
            user = st.session_state.user_data
            
            # Get or select current sentence
            if st.session_state.current_sentence is None:
                st.session_state.current_sentence = select_sentence(
                    df, user, user['rating_10'], skill_window=1
                )
                st.session_state.show_correct_answer = False
                st.session_state.user_answer = ""
            
            current = st.session_state.current_sentence
            
            # Prepare sentence data for JavaScript
            sentence_data = {
                'id': int(current['id']),
                'sentence': str(current['sentence']),
                'translation': str(current['translation']),
                'difficulty': float(current['difficulty']),
                'user_rating': user['rating'],
                'user_rating_10': user['rating_10'],
                'user_rd': user['rd'],
                'cards_learned': len(user['history'])
            }
            
            sentence_json = json.dumps(sentence_data)
            
            # Create fully optimized HTML component with no reruns
            game_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    * {{
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }}
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 0; margin: 0; background: transparent; }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                        gap: 10px;
                        margin-bottom: 15px;
                    }}
                    .stat-card {{
                        background: white;
                        padding: 15px;
                        border-radius: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .stat-label {{
                        font-size: 12px;
                        color: #666;
                        margin-bottom: 6px;
                        font-weight: 500;
                    }}
                    .stat-value {{
                        font-size: 20px;
                        font-weight: 700;
                        color: #333;
                    }}
                    .sentence-card {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 12px;
                        border-radius: 15px;
                        margin-bottom: 15px;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
                    }}
                    .sentence-inner {{
                        background: white;
                        padding: 20px;
                        border-radius: 12px;
                    }}
                    .sentence-meta {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                        justify-content: space-between;
                        font-size: 13px;
                        color: #666;
                        margin-bottom: 15px;
                    }}
                    .sentence-text {{
                        font-size: 20px;
                        font-weight: 500;
                        color: #2d3748;
                        line-height: 1.6;
                    }}
                    
                    /* Tablet and larger screens */
                    @media (min-width: 640px) {{
                        body {{
                            padding: 20px;
                        }}
                        .stats-grid {{
                            gap: 15px;
                            margin-bottom: 20px;
                        }}
                        .stat-card {{
                            padding: 20px;
                        }}
                        .stat-label {{
                            font-size: 14px;
                            margin-bottom: 8px;
                        }}
                        .stat-value {{
                            font-size: 28px;
                        }}
                        .sentence-card {{
                            padding: 20px;
                            margin-bottom: 20px;
                        }}
                        .sentence-inner {{
                            padding: 40px;
                        }}
                        .sentence-meta {{
                            font-size: 14px;
                            margin-bottom: 20px;
                        }}
                        .sentence-text {{
                            font-size: 32px;
                        }}
                    }}
                    
                    /* Large screens */
                    @media (min-width: 1024px) {{
                        .stats-grid {{
                            grid-template-columns: repeat(4, 1fr);
                        }}
                    }}
                    .input-section {{
                        background: white;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        margin-bottom: 15px;
                    }}
                    .input-section.hidden {{
                        display: none;
                    }}
                    .input-group {{
                        margin-bottom: 15px;
                    }}
                    .input-group label {{
                        display: block;
                        font-size: 14px;
                        font-weight: 600;
                        color: #333;
                        margin-bottom: 8px;
                    }}
                    .input-group input {{
                        width: 100%;
                        padding: 12px;
                        font-size: 16px;
                        border: 2px solid #e5e7eb;
                        border-radius: 8px;
                        transition: border-color 0.2s;
                    }}
                    .input-group input:focus {{
                        outline: none;
                        border-color: #667eea;
                    }}
                    .btn {{
                        padding: 12px 20px;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.2s;
                        width: 100%;
                    }}
                    .btn:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    }}
                    .btn-primary {{
                        background: #667eea;
                        color: white;
                    }}
                    .btn-primary:hover {{
                        background: #5568d3;
                    }}
                    .review-section {{
                        background: white;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .review-section.hidden {{
                        display: none;
                    }}
                    .answer-box {{
                        background: #f0fdf4;
                        border: 2px solid #86efac;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 12px;
                    }}
                    .answer-label {{
                        font-size: 12px;
                        font-weight: 600;
                        color: #065f46;
                        margin-bottom: 6px;
                    }}
                    .answer-text {{
                        font-size: 15px;
                        color: #166534;
                        word-wrap: break-word;
                    }}
                    .correct-box {{
                        background: #eff6ff;
                        border: 2px solid #93c5fd;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 15px;
                    }}
                    .correct-label {{
                        font-size: 12px;
                        font-weight: 600;
                        color: #1e40af;
                        margin-bottom: 6px;
                    }}
                    .correct-text {{
                        font-size: 15px;
                        color: #1e3a8a;
                        word-wrap: break-word;
                        line-height: 1.6;
                    }}
                    .eval-title {{
                        font-size: 18px;
                        font-weight: 700;
                        color: #333;
                        margin-bottom: 12px;
                        text-align: center;
                    }}
                    .eval-subtitle {{
                        font-size: 14px;
                        color: #666;
                        margin-bottom: 15px;
                        text-align: center;
                    }}
                    .button-grid {{
                        display: grid;
                        grid-template-columns: 1fr;
                        gap: 10px;
                        margin-bottom: 12px;
                    }}
                    .btn-success {{
                        background: #10b981;
                        color: white;
                    }}
                    .btn-success:hover {{
                        background: #059669;
                    }}
                    .btn-danger {{
                        background: #ef4444;
                        color: white;
                    }}
                    .btn-danger:hover {{
                        background: #dc2626;
                    }}
                    .btn-secondary {{
                        background: #6b7280;
                        color: white;
                    }}
                    .btn-secondary:hover {{
                        background: #4b5563;
                    }}
                    
                    /* Tablet and larger screens - additional styles */
                    @media (min-width: 640px) {{
                        .input-section {{
                            padding: 30px;
                            margin-bottom: 20px;
                        }}
                        .input-group {{
                            margin-bottom: 20px;
                        }}
                        .input-group label {{
                            font-size: 16px;
                            margin-bottom: 10px;
                        }}
                        .input-group input {{
                            padding: 15px;
                            font-size: 18px;
                        }}
                        .btn {{
                            padding: 15px 30px;
                            font-size: 16px;
                        }}
                        .review-section {{
                            padding: 30px;
                        }}
                        .answer-box {{
                            padding: 20px;
                            margin-bottom: 15px;
                        }}
                        .answer-label {{
                            font-size: 14px;
                            margin-bottom: 8px;
                        }}
                        .answer-text {{
                            font-size: 18px;
                        }}
                        .correct-box {{
                            padding: 20px;
                            margin-bottom: 20px;
                        }}
                        .correct-label {{
                            font-size: 14px;
                            margin-bottom: 8px;
                        }}
                        .correct-text {{
                            font-size: 18px;
                        }}
                        .eval-title {{
                            font-size: 20px;
                            margin-bottom: 15px;
                        }}
                        .eval-subtitle {{
                            font-size: 16px;
                            margin-bottom: 20px;
                        }}
                        .button-grid {{
                            grid-template-columns: 1fr 1fr;
                            gap: 15px;
                            margin-bottom: 15px;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Your Skill Level</div>
                        <div class="stat-value" id="skillLevel">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Glicko Rating</div>
                        <div class="stat-value" id="glickoRating">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Rating Deviation</div>
                        <div class="stat-value" id="ratingDev">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Cards Learned</div>
                        <div class="stat-value" id="cardsLearned">-</div>
                    </div>
                </div>
                
                <div class="sentence-card">
                    <div class="sentence-inner">
                        <div class="sentence-meta">
                            <span>🎯 Difficulty: <strong id="difficulty">-</strong>/10</span>
                            <span>📊 Your Level: <strong id="yourLevel">-</strong>/10</span>
                        </div>
                        <div class="sentence-text" id="sentenceText">Loading...</div>
                    </div>
                </div>
                
                <script>
                    const sentenceData = {sentence_json};
                    
                    // Initialize display
                    function init() {{
                        document.getElementById('skillLevel').textContent = sentenceData.user_rating_10.toFixed(2) + '/10';
                        document.getElementById('glickoRating').textContent = Math.round(sentenceData.user_rating);
                        document.getElementById('ratingDev').textContent = sentenceData.user_rd.toFixed(1);
                        document.getElementById('cardsLearned').textContent = sentenceData.cards_learned;
                        
                        document.getElementById('difficulty').textContent = sentenceData.difficulty.toFixed(1);
                        document.getElementById('yourLevel').textContent = sentenceData.user_rating_10.toFixed(2);
                        document.getElementById('sentenceText').textContent = sentenceData.sentence;
                    }}
                    
                    // Initialize on load
                    init();
                </script>
            </body>
            </html>
            """
            
            # Render the component
            components.html(game_html, height=400)
            
            # Text input for answer submission (visible in the interface)
            if not st.session_state.show_correct_answer:
                submitted_answer = st.text_input(
                    "Your Translation:",
                    value="",
                    key="translation_input",
                    placeholder="Type your translation here..."
                )
                
                if st.button("✅ Submit Answer", key="submit_translation", type="primary"):
                    if submitted_answer.strip():
                        st.session_state.user_answer = submitted_answer
                        st.session_state.show_correct_answer = True
                        st.session_state.evaluation_done = False
                        st.session_state.evaluation_result = None
                        st.rerun()
            
            # Show review section if answer was submitted
            if st.session_state.show_correct_answer:
                st.divider()
                
                # Evaluate the translation using AI
                if not st.session_state.evaluation_done:
                    with st.spinner("🤖 Evaluating your answer..."):
                        try:
                            eval_result = evaluate_translation(
                                reference_text=current['sentence'],
                                reference_translation=current['translation'],
                                user_translation=st.session_state.user_answer
                            )
                        except Exception as e:
                            eval_result = {
                                "is_correct": False,
                                "reason": f"Error during evaluation: {str(e)}",
                                "errors": [str(e)]
                            }
                        st.session_state.evaluation_result = eval_result
                        st.session_state.evaluation_done = True
                    
                    # Update user ratings based on AI evaluation
                    sid = int(current['id'])
                    is_correct = eval_result['is_correct']
                    user = update_user_rating(user, current['difficulty'], is_correct)
                    user = update_fsrs_card(user, sid, is_correct)
                    with st.spinner("💾 Saving your progress..."):
                        save_user(user)
                    st.session_state.user_data = user
                
                # Display evaluation results from session state
                eval_result = st.session_state.evaluation_result
                
                st.subheader("📊 Evaluation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📝 Your Answer")
                    st.info(st.session_state.user_answer)
                
                with col2:
                    st.markdown("#### ✅ Reference Translation")
                    st.success(current['translation'])
                
                # Show evaluation verdict
                st.divider()
                
                if eval_result['is_correct']:
                    st.success(f"✅ **Correct!** {eval_result['reason']}")
                else:
                    st.error(f"❌ **Incorrect.** {eval_result['reason']}")
                
                # Show errors if any
                if eval_result.get('errors') and len(eval_result['errors']) > 0:
                    st.markdown("**Errors found:**")
                    for error in eval_result['errors']:
                        st.markdown(f"- {error}")
                
                st.divider()
                
                # Next sentence button (centered)
                if st.button("🔄 Try Another Sentence", type="primary", use_container_width=True, key="next_btn"):
                    st.session_state.current_sentence = None
                    st.session_state.show_correct_answer = False
                    st.session_state.user_answer = ""
                    st.session_state.evaluation_done = False
                    st.session_state.evaluation_result = None
                    st.rerun()


with tab6:
    st.header("📝 Difficulty Annotation")
    st.markdown("""
    Rate each sentence's difficulty on a scale of 1-10 (1=easiest, 10=hardest).
    """)

    if not (st.session_state.processing_complete and st.session_state.processed_results):
        st.warning("⚠️ Please load or process a results CSV before annotating.")
    else:
        # Use shared DataFrame reference
        anno_df = ensure_results_df()
        
        if len(anno_df) == 0:
            st.info("No sentences available to annotate.")
        else:
            # Prepare data for JavaScript
            import json
            
            sentences_data = []
            for idx, row in enumerate(anno_df.itertuples()):
                sentence_obj = {
                    'idx': idx,
                    'sentence': str(row.sentence),
                    'translation': str(getattr(row, 'translation', '')) if pd.notna(getattr(row, 'translation', None)) else '',
                    'transliteration': str(getattr(row, 'transliteration', '')) if pd.notna(getattr(row, 'transliteration', None)) else '',
                    'rh_avg': float(row.rh_avg),
                    'sl': int(row.sl),
                    'start': format_timestamp(getattr(row, 'start', None)),
                    'end': format_timestamp(getattr(row, 'end', None)),
                    'audio_link': st.session_state.get('audio_file_link', ''),
                    'start_time_sec': float(getattr(row, 'start', 0)) if pd.notna(getattr(row, 'start', None)) else 0,
                    'end_time_sec': float(getattr(row, 'end', 0)) if pd.notna(getattr(row, 'end', None)) else 0,
                }
                sentences_data.append(sentence_obj)
            
            annotations_data = {}
            for idx, anno in st.session_state.sentence_annotations.items():
                annotations_data[str(idx)] = {
                    'rating': anno.get('rating', 0),
                    'notes': anno.get('notes', ''),
                    'timestamp': str(anno.get('timestamp', ''))
                }
            
            sentences_json = json.dumps(sentences_data)
            annotations_json = json.dumps(annotations_data)
            
            total_sentences = len(sentences_data)
            annotated_count = len(st.session_state.sentence_annotations)
            
            # Create unified HTML component with annotation interface AND preview section
            unified_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    * {{
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }}
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 0; margin: 0; background: transparent; }}
                    .main-container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    .annotation-panel {{
                        background: white;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                    }}
                    .preview-panel {{
                        background: white;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .nav-buttons {{
                        display: grid;
                        grid-template-columns: repeat(5, 1fr);
                        gap: 10px;
                        margin-bottom: 20px;
                    }}
                    .btn {{
                        padding: 10px 16px;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 500;
                        transition: all 0.2s;
                        background: white;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .btn:hover:not(:disabled) {{
                        transform: translateY(-1px);
                        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
                    }}
                    .btn:disabled {{
                        opacity: 0.5;
                        cursor: not-allowed;
                    }}
                    .sentence-card {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 30px;
                        border-radius: 15px;
                        margin: 20px 0;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }}
                    .sentence-text {{
                        color: white;
                        font-size: 28px;
                        line-height: 1.6;
                        margin: 0;
                    }}
                    .info-grid {{
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .info-box {{
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    }}
                    .info-label {{
                        font-weight: 600;
                        color: #666;
                        margin-bottom: 8px;
                        font-size: 14px;
                    }}
                    .info-content {{
                        color: #333;
                        line-height: 1.5;
                    }}
                    .rating-buttons {{
                        display: grid;
                        grid-template-columns: repeat(4, 1fr);
                        gap: 10px;
                        margin: 20px 0;
                    }}
                    .rating-btn {{
                        padding: 15px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 16px;
                        font-weight: 600;
                        transition: all 0.2s;
                    }}
                    .rating-easy {{
                        background: #10b981;
                        color: white;
                    }}
                    .rating-medium {{
                        background: #f59e0b;
                        color: white;
                    }}
                    .rating-hard {{
                        background: #ef4444;
                        color: white;
                    }}
                    .rating-review {{
                        background: #6b7280;
                        color: white;
                    }}
                    .rating-btn:hover {{
                        transform: scale(1.05);
                    }}
                    .fine-tune {{
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                        display: none;
                    }}
                    .fine-tune.show {{
                        display: block;
                    }}
                    .toggle-btn {{
                        width: 100%;
                        padding: 12px;
                        background: #3b82f6;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 600;
                        margin-bottom: 15px;
                    }}
                    .input-group {{
                        margin: 15px 0;
                    }}
                    .input-group label {{
                        display: block;
                        margin-bottom: 5px;
                        font-weight: 600;
                        color: #333;
                    }}
                    .input-group input, .input-group textarea {{
                        width: 100%;
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                        font-size: 14px;
                    }}
                    .status-indicator {{
                        display: inline-block;
                        padding: 6px 12px;
                        border-radius: 4px;
                        font-size: 14px;
                        font-weight: 600;
                        margin-bottom: 10px;
                    }}
                    .status-annotated {{
                        background: #d1fae5;
                        color: #065f46;
                    }}
                    .status-unannotated {{
                        background: #fee2e2;
                        color: #991b1b;
                    }}
                    .audio-player {{
                        margin: 15px 0;
                        padding: 15px;
                        background: #f9fafb;
                        border-radius: 8px;
                    }}
                    .custom-audio-controls {{
                        display: flex;
                        align-items: center;
                        gap: 15px;
                        margin-top: 10px;
                    }}
                    .play-pause-btn {{
                        width: 50px;
                        height: 50px;
                        border-radius: 50%;
                        border: none;
                        background: #3b82f6;
                        color: white;
                        font-size: 20px;
                        cursor: pointer;
                        transition: all 0.2s;
                    }}
                    .play-pause-btn:hover {{
                        background: #2563eb;
                        transform: scale(1.05);
                    }}
                    .progress-container {{
                        flex: 1;
                        height: 8px;
                        background: #e5e7eb;
                        border-radius: 4px;
                        cursor: pointer;
                        position: relative;
                    }}
                    .progress-bar {{
                        height: 100%;
                        background: #3b82f6;
                        border-radius: 4px;
                        width: 0%;
                        transition: width 0.1s;
                    }}
                    .time-display {{
                        font-size: 14px;
                        color: #666;
                        min-width: 100px;
                        text-align: right;
                    }}
                    #masterAudio {{
                        display: none;
                    }}
                    .save-info {{
                        background: #eff6ff;
                        padding: 10px;
                        border-radius: 6px;
                        margin-top: 10px;
                        font-size: 13px;
                        color: #1e40af;
                    }}
                    
                    /* Preview panel styles */
                    .preview-header {{
                        font-size: 18px;
                        font-weight: 700;
                        margin-bottom: 15px;
                        color: #333;
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: 1fr;
                        gap: 10px;
                        margin: 15px 0;
                    }}
                    .stat-box {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 15px;
                        border-radius: 8px;
                        color: white;
                    }}
                    .stat-label {{
                        font-size: 12px;
                        opacity: 0.9;
                        margin-bottom: 5px;
                        font-weight: 500;
                    }}
                    .stat-value {{
                        font-size: 24px;
                        font-weight: 700;
                    }}
                    .info-banner {{
                        background: #eff6ff;
                        padding: 12px;
                        border-radius: 6px;
                        border-left: 4px solid #3b82f6;
                        margin: 10px 0;
                        font-size: 13px;
                        color: #1e40af;
                    }}
                    .export-section {{
                        margin-top: 15px;
                        padding-top: 15px;
                        border-top: 2px solid #e5e7eb;
                    }}
                    .btn-primary {{
                        background: #3b82f6;
                        color: white;
                    }}
                    .btn-primary:hover {{
                        background: #2563eb;
                    }}
                    .btn-success {{
                        background: #10b981;
                        color: white;
                    }}
                    .btn-success:hover {{
                        background: #059669;
                    }}
                    #downloadLink {{
                        display: none;
                    }}
                </style>
            </head>
            <body>
                <div class="main-container">
                    <!-- Left Panel: Annotation Interface -->
                    <div class="annotation-panel">
                    <div class="nav-buttons">
                        <button class="btn" onclick="goFirst()" id="btnFirst">⏮️ First</button>
                        <button class="btn" onclick="goPrev()" id="btnPrev">◀️ Previous</button>
                        <button class="btn" onclick="goNext()" id="btnNext">▶️ Next</button>
                        <button class="btn" onclick="goLast()" id="btnLast">⏭️ Last</button>
                        <button class="btn" onclick="goUnannotated()">🎯 Next Unannotated</button>
                    </div>
                    
                    <div id="sentenceDisplay"></div>
                    
                    <div class="info-grid" id="infoGrid"></div>
                    
                    <div class="audio-player" id="audioContainer" style="display:none;">
                        <div class="info-label">🔊 Audio Playback</div>
                        <div class="custom-audio-controls">
                            <button class="play-pause-btn" id="playPauseBtn" onclick="togglePlayPause()">▶️</button>
                            <div class="progress-container" id="progressContainer" onclick="seekAudio(event)">
                                <div class="progress-bar" id="progressBar"></div>
                            </div>
                            <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
                        </div>
                    </div>
                    
                    <div>
                        <div class="input-group">
                            <label>Rating (1-10):</label>
                            <input type="number" id="ratingInput" min="1" max="10" value="5">
                        </div>
                        <div style="display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 10px; margin-top: 15px;">
                            <button class="btn" style="background: #3b82f6; color: white;" onclick="saveAndNext()">💾 Save & Next</button>
                            <button class="btn" style="background: #10b981; color: white;" onclick="save()">💾 Save</button>
                            <button class="btn" style="background: #ef4444; color: white;" onclick="clearAnnotation()">🗑️ Clear</button>
                        </div>
                    </div>
                    
                        <div class="save-info" id="saveInfo" style="display:none;"></div>
                    </div>
                    
                    <!-- Right Panel: Preview & Export -->
                    <div class="preview-panel">
                        <div class="preview-header">💾 Export & Statistics</div>
                        
                        <div class="info-banner">
                            ✨ Updates automatically after each save!
                        </div>
                        
                        <div class="stats-grid">
                            <div class="stat-box">
                                <div class="stat-label">Annotated</div>
                                <div class="stat-value" id="annotatedCount">0</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Avg Rating</div>
                                <div class="stat-value" id="avgRating">0.0</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Easy/Med/Hard</div>
                                <div class="stat-value" id="distribution">0/0/0</div>
                            </div>
                        </div>
                        
                        <div class="export-section">
                            <button class="btn btn-primary" onclick="prepareExport()" id="prepareBtn">
                                📥 Prepare CSV Export
                            </button>
                            <div id="downloadContainer" style="display: none; margin-top: 10px;">
                                <button class="btn btn-success" onclick="downloadCSV()" id="downloadBtn">
                                    ⬇️ Download Annotated CSV
                                </button>
                            </div>
                            <a id="downloadLink" download=""></a>
                        </div>
                    </div>
                </div>
                
                <script>
                    const sentences = {sentences_json};
                    const annotations = {annotations_json};
                    let currentIdx = 0;
                    
                    // Master audio element and playback state - use global master audio
                    // Access global master audio - try multiple levels up the window hierarchy
                    // Use a getter function to always fetch the latest reference
                    function getMasterAudio() {{
                        return (window.parent.parent && window.parent.parent.globalMasterAudio) || 
                               (window.parent && window.parent.globalMasterAudio) || 
                               (window.top && window.top.globalMasterAudio);
                    }}
                    let currentStartTime = 0;
                    let currentEndTime = 0;
                    let snippetInterval = null;
                    let isPlaying = false;
                    
                    // Update progress bar and time display
                    function updateProgress() {{
                        const masterAudio = getMasterAudio();
                        if (!masterAudio || currentEndTime <= currentStartTime) return;
                        
                        const currentTime = masterAudio.currentTime;
                        const duration = currentEndTime - currentStartTime;
                        const elapsed = Math.max(0, Math.min(currentTime - currentStartTime, duration));
                        const progress = (elapsed / duration) * 100;
                        
                        document.getElementById('progressBar').style.width = progress + '%';
                        
                        const formatTime = (seconds) => {{
                            const mins = Math.floor(seconds / 60);
                            const secs = Math.floor(seconds % 60);
                            return mins + ':' + (secs < 10 ? '0' : '') + secs;
                        }};
                        
                        document.getElementById('timeDisplay').textContent = 
                            formatTime(elapsed) + ' / ' + formatTime(duration);
                    }}
                    
                    // Play/Pause toggle
                    function togglePlayPause() {{
                        if (isPlaying) {{
                            pauseAudio();
                        }} else {{
                            playAudio();
                        }}
                    }}
                    
                    // Play audio snippet
                    function playAudio() {{
                        const masterAudio = getMasterAudio();
                        if (!masterAudio) return;
                        
                        // Set to start time if outside the snippet range
                        if (masterAudio.currentTime < currentStartTime || masterAudio.currentTime >= currentEndTime) {{
                            masterAudio.currentTime = currentStartTime;
                        }}
                        
                        masterAudio.play();
                        isPlaying = true;
                        document.getElementById('playPauseBtn').textContent = '⏸️';
                        
                        // Clear any existing interval
                        if (snippetInterval) {{
                            clearInterval(snippetInterval);
                        }}
                        
                        // Monitor playback and stop at end time
                        snippetInterval = setInterval(() => {{
                            updateProgress();
                            
                            if (masterAudio.currentTime >= currentEndTime) {{
                                pauseAudio();
                                masterAudio.currentTime = currentStartTime;
                                updateProgress();
                            }}
                        }}, 50);
                    }}
                    
                    // Pause audio
                    function pauseAudio() {{
                        const masterAudio = getMasterAudio();
                        if (!masterAudio) return;
                        
                        masterAudio.pause();
                        isPlaying = false;
                        document.getElementById('playPauseBtn').textContent = '▶️';
                        
                        if (snippetInterval) {{
                            clearInterval(snippetInterval);
                            snippetInterval = null;
                        }}
                    }}
                    
                    // Seek in audio
                    function seekAudio(event) {{
                        const masterAudio = getMasterAudio();
                        if (!masterAudio || currentEndTime <= currentStartTime) return;
                        
                        const progressContainer = document.getElementById('progressContainer');
                        const rect = progressContainer.getBoundingClientRect();
                        const clickX = event.clientX - rect.left;
                        const percentage = clickX / rect.width;
                        
                        const duration = currentEndTime - currentStartTime;
                        const newTime = currentStartTime + (duration * percentage);
                        
                        masterAudio.currentTime = Math.max(currentStartTime, Math.min(newTime, currentEndTime));
                        updateProgress();
                    }}
                    
                    // Load snippet for current sentence
                    function loadAudioSnippet(startTime, endTime) {{
                        const masterAudio = getMasterAudio();
                        currentStartTime = startTime;
                        currentEndTime = endTime;
                        
                        // Reset playback state
                        pauseAudio();
                        if (masterAudio) {{
                            masterAudio.currentTime = startTime;
                        }}
                        updateProgress();
                    }}
                    
                    function render() {{
                        if (sentences.length === 0) return;
                        
                        const sentence = sentences[currentIdx];
                        const annotation = annotations[currentIdx] || {{}};
                        
                        // Update navigation buttons
                        document.getElementById('btnFirst').disabled = currentIdx === 0;
                        document.getElementById('btnPrev').disabled = currentIdx === 0;
                        document.getElementById('btnNext').disabled = currentIdx >= sentences.length - 1;
                        document.getElementById('btnLast').disabled = currentIdx >= sentences.length - 1;
                        
                        // Status indicator
                        const status = annotations[currentIdx] ? 
                            '<span class="status-annotated">✅ Annotated</span>' :
                            '<span class="status-unannotated">⭕ Not Annotated</span>';
                        
                        // Sentence display
                        document.getElementById('sentenceDisplay').innerHTML = `
                            <div style="margin-bottom: 10px;">
                                ${{status}}
                                <span style="margin-left: 10px; font-weight: 600;">Sentence ${{currentIdx + 1}} of ${{sentences.length}}</span>
                            </div>
                            <div class="sentence-card">
                                <h2 class="sentence-text">${{escapeHtml(sentence.sentence)}}</h2>
                            </div>
                        `;
                        
                        // Info grid
                        let infoHTML = '';
                        if (sentence.translation) {{
                            infoHTML += `
                                <div class="info-box">
                                    <div class="info-label">🌐 Translation</div>
                                    <div class="info-content">${{escapeHtml(sentence.translation)}}</div>
                                </div>
                            `;
                        }}
                        if (sentence.transliteration) {{
                            infoHTML += `
                                <div class="info-box">
                                    <div class="info-label">🔤 Transliteration</div>
                                    <div class="info-content">${{escapeHtml(sentence.transliteration)}}</div>
                                </div>
                            `;
                        }}
                        infoHTML += `
                            <div class="info-box">
                                <div class="info-label">📊 Metrics</div>
                                <div class="info-content">
                                    RH Avg: ${{sentence.rh_avg.toFixed(1)}}<br>
                                    Length: ${{sentence.sl}} words<br>
                                    Time: ${{sentence.start}} → ${{sentence.end}}
                                </div>
                            </div>
                        `;
                        document.getElementById('infoGrid').innerHTML = infoHTML;
                        
                        // Audio player - load snippet times
                        if (sentence.audio_link && getMasterAudio()) {{
                            document.getElementById('audioContainer').style.display = 'block';
                            loadAudioSnippet(sentence.start_time_sec, sentence.end_time_sec);
                        }} else {{
                            document.getElementById('audioContainer').style.display = 'none';
                        }}
                        
                        // Load existing annotation
                        if (annotation.rating) {{
                            document.getElementById('ratingInput').value = annotation.rating;
                            document.getElementById('saveInfo').innerHTML = `✅ Saved: Rating ${{annotation.rating}}/10`;
                            document.getElementById('saveInfo').style.display = 'block';
                        }} else {{
                            document.getElementById('ratingInput').value = 5;
                            document.getElementById('saveInfo').style.display = 'none';
                        }}
                    }}
                    
                    function escapeHtml(text) {{
                        const div = document.createElement('div');
                        div.textContent = text;
                        return div.innerHTML;
                    }}
                    
                    function goFirst() {{
                        currentIdx = 0;
                        render();
                    }}
                    
                    function goPrev() {{
                        if (currentIdx > 0) {{
                            currentIdx--;
                            render();
                        }}
                    }}
                    
                    function goNext() {{
                        if (currentIdx < sentences.length - 1) {{
                            currentIdx++;
                            render();
                        }}
                    }}
                    
                    function goLast() {{
                        currentIdx = sentences.length - 1;
                        render();
                    }}
                    
                    function goUnannotated() {{
                        let found = false;
                        for (let i = currentIdx + 1; i < sentences.length; i++) {{
                            if (!annotations[i]) {{
                                currentIdx = i;
                                found = true;
                                break;
                            }}
                        }}
                        if (!found) {{
                            for (let i = 0; i < currentIdx; i++) {{
                                if (!annotations[i]) {{
                                    currentIdx = i;
                                    found = true;
                                    break;
                                }}
                            }}
                        }}
                        render();
                    }}
                    
                    function save() {{
                        const rating = parseInt(document.getElementById('ratingInput').value);
                        
                        annotations[currentIdx] = {{
                            rating: rating
                        }};
                        
                        window.parent.postMessage({{
                            type: 'annotation',
                            idx: currentIdx,
                            data: annotations[currentIdx]
                        }}, '*');
                        
                        // Update preview immediately after save
                        updatePreviewStats();
                        render();
                    }}
                    
                    function saveAndNext() {{
                        save();
                        if (currentIdx < sentences.length - 1) {{
                            currentIdx++;
                            render();
                        }}
                    }}
                    
                    function clearAnnotation() {{
                        delete annotations[currentIdx];
                        
                        window.parent.postMessage({{
                            type: 'clear_annotation',
                            idx: currentIdx
                        }}, '*');
                        
                        // Update preview after clearing
                        updatePreviewStats();
                        render();
                    }}
                    
                    // Function to update preview stats in real-time
                    function updatePreviewStats() {{
                        const annotatedCount = Object.keys(annotations).length;
                        let ratings = [];
                        let easy = 0, medium = 0, hard = 0;
                        
                        for (let key in annotations) {{
                            const rating = annotations[key].rating || 0;
                            if (rating > 0) {{
                                ratings.push(rating);
                                if (rating >= 1 && rating <= 3) easy++;
                                else if (rating >= 4 && rating <= 6) medium++;
                                else if (rating >= 7 && rating <= 10) hard++;
                            }}
                        }}
                        
                        const avgRating = ratings.length > 0 ? 
                            (ratings.reduce((a, b) => a + b, 0) / ratings.length).toFixed(1) : '0.0';
                        
                        // Update preview panel stats
                        document.getElementById('annotatedCount').textContent = annotatedCount;
                        document.getElementById('avgRating').textContent = avgRating;
                        document.getElementById('distribution').textContent = easy + '/' + medium + '/' + hard;
                        
                        // Show/hide export button based on annotations
                        if (annotatedCount > 0) {{
                            document.getElementById('prepareBtn').style.display = 'block';
                        }}
                    }}
                    
                    function prepareExport() {{
                        document.getElementById('downloadContainer').style.display = 'block';
                        document.getElementById('prepareBtn').textContent = '✅ Export Ready';
                    }}
                    
                    function downloadCSV() {{
                        let csvContent = '';
                        const headers = Object.keys(sentences[0]);
                        csvContent += headers.join(',') + ',difficulty_rating\\n';
                        
                        sentences.forEach((sentence, idx) => {{
                            const row = [];
                            headers.forEach(header => {{
                                let value = sentence[header];
                                if (typeof value === 'string') {{
                                    value = value.replace(/"/g, '""');
                                    if (value.includes(',') || value.includes('\\n')) {{
                                        value = '"' + value + '"';
                                    }}
                                }}
                                row.push(value);
                            }});
                            
                            const anno = annotations[idx] || {{}};
                            row.push(anno.rating || '');
                            
                            csvContent += row.join(',') + '\\n';
                        }});
                        
                        const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                        const url = URL.createObjectURL(blob);
                        const link = document.getElementById('downloadLink');
                        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                        link.href = url;
                        link.download = 'annotated_' + timestamp + '.csv';
                        link.click();
                        URL.revokeObjectURL(url);
                        
                        document.getElementById('downloadBtn').textContent = '✅ Downloaded!';
                        setTimeout(() => {{
                            document.getElementById('downloadBtn').textContent = '⬇️ Download Annotated CSV';
                        }}, 2000);
                    }}
                    
                    // Initial render
                    render();
                    
                    // Initial preview update
                    updatePreviewStats();
                </script>
            </body>
            </html>
            """
            
            # Render the unified HTML component
            components.html(unified_html, height=900, scrolling=True)
        
with tab7:
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