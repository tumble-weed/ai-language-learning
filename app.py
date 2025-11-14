"""
Streamlit frontend for Language Learning Difficulty Analyzer
Provides audio upload, processing pipeline, and interactive results visualization.
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict
import os

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Load Results", "📤 Upload & Process", "📊 Results", "🎯 Practice", "ℹ️ About"])

with tab1:
    st.header("Load Existing Results")
    
    st.markdown("""
    Skip the processing and directly load results from a previously generated CSV file.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader for CSV
        results_file = st.file_uploader(
            "Upload Results CSV File",
            type=['csv'],
            help="Upload a CSV file with analysis results"
        )
    
    with col2:
        st.info("💡 **Quick Start**\n\nUpload a results CSV to view analysis without processing.")
    
    if results_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(results_file)
            
            # Column mapping for different CSV formats
            column_mapping = {
                'Audio_file': 'audio_file',
                'Text': 'sentence',
                'RH1_Result': 'rh1',
                'RH2_Result': 'rh2',
                'RH_Average': 'rh_avg',
                'SL_Result': 'sl',
                'Chunk_ID': 'audio_file',  # Fallback if no Audio_file
                'WFR_Result': 'wfr',
                'Transliteration': 'transliteration',
                'Translation': 'translation'
            }
            
            # Rename columns to standardized format
            df = df.rename(columns=column_mapping)
            
            # Check if we have required columns after mapping
            required_cols = ['sentence', 'rh1', 'rh2', 'rh_avg', 'sl']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # Handle missing audio_file column
            if 'audio_file' not in df.columns:
                df['audio_file'] = 'N/A'
                st.warning("⚠️ CSV doesn't have audio file column. Audio playback will not be available.")
            
            if missing_cols:
                st.error(f"❌ CSV is missing required columns: {', '.join(missing_cols)}")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
                st.info("Expected: Text, RH1_Result, RH2_Result, RH_Average, SL_Result (and optionally Audio_file)")
            else:
                # Add missing optional columns with defaults
                if 'start_time' not in df.columns:
                    df['start_time'] = 0
                if 'end_time' not in df.columns:
                    df['end_time'] = 0
                if 'wfr' not in df.columns:
                    df['wfr'] = None
                if 'transliteration' not in df.columns:
                    df['transliteration'] = None
                if 'translation' not in df.columns:
                    df['translation'] = None
                
                # Convert DataFrame to list of dicts for session state
                results_list = df.to_dict('records')
                st.session_state.processed_results = results_list
                st.session_state.processing_complete = True
                
                st.success(f"✅ Successfully loaded {len(results_list)} sentences!")
                st.info("👉 Go to the **Results** tab to view and interact with the data")
                
                # Show preview
                st.subheader("Preview (First 5 Rows)")
                preview_df = df.head(5)[['sentence', 'rh_avg', 'sl']]
                st.dataframe(preview_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.exception(e)
    
    st.divider()
    
    # Quick access to recent results
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
            
            if selected_file != "-- Select a file --":
                if st.button("📥 Load Selected File", type="primary"):
                    try:
                        file_path = csv_options[selected_file]
                        df = pd.read_csv(file_path)
                        
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
                            'Translation': 'translation'
                        }
                        
                        # Rename columns
                        df = df.rename(columns=column_mapping)
                        
                        # Validate and load
                        required_cols = ['sentence', 'rh1', 'rh2', 'rh_avg', 'sl']
                        
                        # Handle missing audio_file
                        if 'audio_file' not in df.columns:
                            df['audio_file'] = 'N/A'
                        
                        # Add defaults for optional columns
                        if 'start_time' not in df.columns:
                            df['start_time'] = 0
                        if 'end_time' not in df.columns:
                            df['end_time'] = 0
                        if 'transliteration' not in df.columns:
                            df['transliteration'] = None
                        if 'translation' not in df.columns:
                            df['translation'] = None
                        
                        if all(col in df.columns for col in required_cols):
                            results_list = df.to_dict('records')
                            st.session_state.processed_results = results_list
                            st.session_state.processing_complete = True
                            
                            st.success(f"✅ Loaded {len(results_list)} sentences from {file_path.name}!")
                            st.info("👉 Go to the **Results** tab to view the data")
                        else:
                            missing = [col for col in required_cols if col not in df.columns]
                            st.error(f"❌ Selected file is missing columns: {', '.join(missing)}")
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
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
            BASE_DIR = Path(__file__).resolve().parent
            OUTPUT_DIR = BASE_DIR / "output" / "streamlit_temp"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            TRANSCRIBE_OUTPUT_FILE = OUTPUT_DIR / "transcribed_output.pkl"
            SENTENCE_OUTPUT_FILE = OUTPUT_DIR / "sentences_output.txt"
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Segmentation
                status_text.text("🔊 Segmenting audio into dialogue chunks...")
                progress_bar.progress(10)
                
                exported_chunk_paths = segment_dialogue(
                    audio_file_path=st.session_state.audio_file_path,
                    output_dir=OUTPUT_DIR / "chunks",
                    vad_aggressiveness=vad_aggressiveness,
                    min_speech_len_ms=min_speech_len_ms,
                    min_silence_after_speech_ms=min_silence_ms
                )
                
                if not exported_chunk_paths:
                    st.error("❌ No dialogue segments found. Try adjusting VAD settings.")
                    st.stop()
                
                progress_bar.progress(30)
                st.info(f"Found {len(exported_chunk_paths)} dialogue segments")
                
                # Step 2: Transcription
                status_text.text("📝 Transcribing audio chunks...")
                progress_bar.progress(40)
                
                transcribed_text = indic_transcribe_chunks(
                    lang_code=lang_code,
                    exported_chunk_paths=exported_chunk_paths,
                    output_file=TRANSCRIBE_OUTPUT_FILE
                )
                
                progress_bar.progress(60)
                
                # Step 3: Load transcription and prepare sentences
                status_text.text("🔄 Processing transcriptions...")
                with TRANSCRIBE_OUTPUT_FILE.open('rb') as f_in:
                    transcribed_data = pickle.load(f_in)
                
                # For this simplified version, use transcribed text as sentences
                # (In full pipeline, you'd apply punctuation restoration and preprocessing)
                preprocessed_text = [transcript['text'] for transcript in transcribed_data]
                
                with SENTENCE_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
                    f_out.write("\n".join(preprocessed_text))
                
                progress_bar.progress(70)
                
                # Step 4: Align sentences to timestamps
                status_text.text("⏱️ Aligning sentences to timestamps...")
                aligned_result = align_sentences_to_timestamps(
                    transcribed_data,
                    preprocessed_text,
                    st.session_state.audio_file_path
                )
                
                progress_bar.progress(80)
                
                # Step 5: Calculate metrics
                status_text.text("📊 Calculating difficulty metrics...")
                rh1 = IndicReadabilityRH1()
                rh2 = IndicReadabilityRH2()
                sl = SentenceLengthMetric()
                
                results_list = []
                for t in aligned_result:
                    rh1Res = rh1.compute(t["sentence"])
                    rh2Res = rh2.compute(t["sentence"])
                    avg = (rh1Res + rh2Res) / 2
                    slRes = sl.compute(t["sentence"])
                    
                    row_data = {
                        "audio_file": t["audio_file"],
                        "sentence": t["sentence"],
                        "start_time": t.get("start_time", 0),
                        "end_time": t.get("end_time", 0),
                        "rh1": rh1Res,
                        "rh2": rh2Res,
                        "rh_avg": avg,
                        "sl": slRes
                    }
                    results_list.append(row_data)
                
                st.session_state.processed_results = results_list
                st.session_state.processing_complete = True
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.success(f"🎉 Successfully processed {len(results_list)} sentences!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)

with tab3:
    st.header("Analysis Results")
    
    if st.session_state.processing_complete and st.session_state.processed_results:
        df = pd.DataFrame(st.session_state.processed_results)
        
        # Apply sorting
        if sort_by == 'Difficulty (Easy → Hard)':
            df = df.sort_values(['sl', 'rh_avg'], ascending=[True, True])
        elif sort_by == 'Difficulty (Hard → Easy)':
            df = df.sort_values(['sl', 'rh_avg'], ascending=[False, False])
        elif sort_by == 'Sentence Length':
            df = df.sort_values('sl', ascending=True)
        else:  # Audio File
            df = df.sort_values('audio_file', ascending=True)
        
        # Reset index for display
        df = df.reset_index(drop=True)
        
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
                value=(int(df['sl'].min()), int(df['sl'].max()))
            )
        
        with col2:
            rh_range = st.slider(
                "RH Average Range",
                min_value=float(df['rh_avg'].min()),
                max_value=float(df['rh_avg'].max()),
                value=(float(df['rh_avg'].min()), float(df['rh_avg'].max())),
                step=0.1
            )
        
        # Apply filters
        filtered_df = df[
            (df['sl'] >= sl_range[0]) & 
            (df['sl'] <= sl_range[1]) &
            (df['rh_avg'] >= rh_range[0]) & 
            (df['rh_avg'] <= rh_range[1])
        ]
        
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
                            
                            # Sentence text in a highlighted box
                            st.markdown(f"""
                            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong>{row['sentence']}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Transliteration below text if available
                            if row.get('transliteration') and pd.notna(row['transliteration']):
                                st.markdown(f"""
                                <div style='background-color: #e8f4f8; padding: 8px; border-radius: 5px; margin-bottom: 10px; font-style: italic; color: #555;'>
                                    {row['transliteration']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Translation below transliteration if available
                            if row.get('translation') and pd.notna(row['translation']):
                                st.markdown(f"""
                                <div style='background-color: #fff4e6; padding: 8px; border-radius: 5px; margin-bottom: 10px; color: #333;'>
                                    {row['translation']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Audio playback
                            audio_path = Path(row['audio_file'])
                            if audio_path.exists():
                                st.audio(str(audio_path), format='audio/wav')
                            else:
                                st.warning("⚠️ Audio not found")
                            
                            # Metrics in compact layout
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Length", f"{row['sl']} words", label_visibility="visible")
                                st.metric("RH1", f"{row['rh1']:.2f}", label_visibility="visible")
                            with metric_col2:
                                st.metric("RH2", f"{row['rh2']:.2f}", label_visibility="visible")
                                st.metric("Avg", f"{row['rh_avg']:.2f}", label_visibility="visible")
                            
                            st.markdown("---")
        else:
            # Original list view with expanders
            for idx, row in filtered_df.iterrows():
                with st.expander(f"**Sentence {idx+1}**: {row['sentence'][:50]}..." if len(row['sentence']) > 50 else f"**Sentence {idx+1}**: {row['sentence']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Full Text:** {row['sentence']}")
                        
                        # Transliteration below text if available
                        if row.get('transliteration') and pd.notna(row['transliteration']):
                            st.markdown(f"**Transliteration:** *{row['transliteration']}*")
                        
                        # Translation below transliteration if available
                        if row.get('translation') and pd.notna(row['translation']):
                            st.markdown(f"**Translation:** {row['translation']}")
                        
                        st.markdown(f"**Audio File:** `{Path(row['audio_file']).name}`")
                        
                        # Audio playback if file exists
                        audio_path = Path(row['audio_file'])
                        if audio_path.exists():
                            st.audio(str(audio_path), format='audio/wav')
                        else:
                            st.warning("⚠️ Audio file not found")
                    
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
    Practice your language skills! See the translation and try to guess the original sentence, 
    then reveal and play the audio to check your answer.
    """)
    
    if st.session_state.processing_complete and st.session_state.processed_results:
        df = pd.DataFrame(st.session_state.processed_results)
        
        # Filter to only sentences that have translations
        practice_df = df[df['translation'].notna() & (df['translation'] != '')]
        
        if len(practice_df) == 0:
            st.warning("⚠️ No sentences with translations available for practice. Please load a CSV file with a Translation column.")
        else:
            # Initialize practice session state
            if 'practice_index' not in st.session_state:
                st.session_state.practice_index = 0
            if 'show_answer' not in st.session_state:
                st.session_state.show_answer = False
            if 'practice_order' not in st.session_state:
                # Shuffle for random practice
                st.session_state.practice_order = practice_df.sample(frac=1).reset_index(drop=True)
            
            practice_sentences = st.session_state.practice_order
            current_idx = st.session_state.practice_index
            
            if current_idx >= len(practice_sentences):
                st.success("🎉 You've completed all sentences! Click 'Restart Practice' to go again.")
                if st.button("🔄 Restart Practice", type="primary"):
                    st.session_state.practice_index = 0
                    st.session_state.show_answer = False
                    st.session_state.practice_order = practice_df.sample(frac=1).reset_index(drop=True)
                    st.rerun()
            else:
                current_sentence = practice_sentences.iloc[current_idx]
                
                # Progress indicator
                st.progress((current_idx + 1) / len(practice_sentences))
                st.caption(f"Sentence {current_idx + 1} of {len(practice_sentences)}")
                
                st.divider()
                
                # Display translation in a prominent box
                st.markdown("### 📖 Translation:")
                st.markdown(f"""
                <div style='background-color: #fff4e6; padding: 20px; border-radius: 10px; margin: 20px 0; font-size: 1.2em; text-align: center;'>
                    <strong>{current_sentence['translation']}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Difficulty hint
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💡 Difficulty", f"{current_sentence['rh_avg']:.1f}")
                with col2:
                    st.metric("📏 Length", f"{int(current_sentence['sl'])} words")
                with col3:
                    difficulty_label = "Easy" if current_sentence['rh_avg'] < 33 else "Medium" if current_sentence['rh_avg'] < 66 else "Hard"
                    st.metric("🎯 Level", difficulty_label)
                
                st.divider()
                
                # User input area
                st.markdown("### ✍️ Your Guess:")
                user_guess = st.text_area(
                    "Type what you think the original sentence is:",
                    height=100,
                    key=f"guess_{current_idx}",
                    placeholder="Type your answer here..."
                )
                
                # Control buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("👁️ Show Answer", type="primary", use_container_width=True):
                        st.session_state.show_answer = True
                
                with col2:
                    if st.button("⏭️ Next Sentence", use_container_width=True):
                        st.session_state.practice_index += 1
                        st.session_state.show_answer = False
                        st.rerun()
                
                # Show answer section
                if st.session_state.show_answer:
                    st.divider()
                    st.markdown("### ✅ Correct Answer:")
                    
                    # Original sentence
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                        <strong style='font-size: 1.1em;'>{current_sentence['sentence']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Transliteration if available
                    if current_sentence.get('transliteration') and pd.notna(current_sentence['transliteration']):
                        st.markdown(f"""
                        <div style='background-color: #e8f4f8; padding: 12px; border-radius: 5px; margin-bottom: 10px; font-style: italic; color: #555;'>
                            {current_sentence['transliteration']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Audio playback
                    st.markdown("#### 🔊 Listen:")
                    audio_path = Path(current_sentence['audio_file'])
                    if audio_path.exists():
                        st.audio(str(audio_path), format='audio/wav')
                    else:
                        st.warning("⚠️ Audio file not found")
                    
                    # Comparison if user typed something
                    if user_guess.strip():
                        st.markdown("#### 📝 Your Answer vs Correct Answer:")
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.markdown("**Your Guess:**")
                            st.info(user_guess)
                        with comp_col2:
                            st.markdown("**Correct:**")
                            st.success(current_sentence['sentence'])
            
            # Practice controls at bottom
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Shuffle & Restart", use_container_width=True):
                    st.session_state.practice_index = 0
                    st.session_state.show_answer = False
                    st.session_state.practice_order = practice_df.sample(frac=1).reset_index(drop=True)
                    st.rerun()
            
            with col2:
                if st.button("⏮️ Previous", disabled=(current_idx == 0), use_container_width=True):
                    st.session_state.practice_index = max(0, current_idx - 1)
                    st.session_state.show_answer = False
                    st.rerun()
            
            with col3:
                st.caption(f"Progress: {min(current_idx + 1, len(practice_sentences))}/{len(practice_sentences)}")
    
    else:
        st.info("👈 Load results from the **Load Results** tab to start practicing!")

with tab5:
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
