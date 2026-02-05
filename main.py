import argparse
import sys
import os
from utils.yt_wav_downloader import download_youtube_as_wav
from segmentation.webrtc_dialogue_segmentation import segment_dialogue
# from transcription.whisper_chunker import whisper_transcribe_chunks
from transcription.indic_chunker import indic_transcribe_chunks
from restore_punctuation.indic_punc_resto import restore_punctuation
from preprocessing.indic_preprocessor import preprocess_text
from sentence_alignment.align_sentences import align_sentences_to_timestamps
from transliteration.indic_en_trlit import transliterate_indic_to_english
from translation.indic_en_translation import translate_indic_to_english
from utils.rclone_helper import upload_files
from metrics.english_metrics import get_features
# from metrics.hindi_models import IndicReadabilityRH1, IndicReadabilityRH2
# from metrics.wfr import WordFrequencyMetric
# from metrics.sl import SentenceLengthMetric
# import csv
from pathlib import Path
import dotenv
import pickle
import pandas as pd
import joblib
import json
import traceback
sys.stdout.reconfigure(encoding='utf-8')

dotenv.load_dotenv()  # Load environment variables from .env file


BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE_DIR = BASE_DIR / "input"
CONTINUE_FROM = 'download'  # Default step to start from
YOUTUBE_LINK = None
AUDIO_FILE_PATH = None  # Initialize AUDIO_FILE_PATH
link = None

STEPS = [
    'download',
    'segment',
    'transcribe',
    'punctuate',
    'preprocess',
    'align',
    'transliterate',
    'translate',
    'metrics',
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Language Learning Pipeline.")
    parser.add_argument(
        '--continue-from',
        type=str,
        choices=STEPS,
        default='download',
        help=f'Continue execution from a specific step. Choose from: {", ".join(STEPS)}'
    )

    parser.add_argument(
        '--i',
        type=str,
        default=None,
        help='Input json file for initializing parameters.'
    )

    parser.add_argument(
        '--yt-link',
        type=str,
        default=None,
        help='YouTube link of the video to be processed.'
    )

    return parser.parse_args()

def get_step_index(step_name):
    return STEPS.index(step_name)

def should_execute(current_step, continue_from):
    return get_step_index(current_step) >= get_step_index(continue_from)


args = parse_arguments()
CONTINUE_FROM = args.continue_from
INPUT_FILE = args.i
YOUTUBE_LINK = args.yt_link

if not YOUTUBE_LINK:
    print("ERROR: YouTube link must be provided via --yt-link argument.")
    sys.exit(1)

if INPUT_FILE:
    # Load parameters from the input JSON file
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            params = json.load(f)
            AUDIO_FILE_PATH = Path(params.get('audio_file_path', None))
            DROPBOX_LINK = params.get('dropbox_link', None)
    except Exception as e:
        print(f"Error reading input file {INPUT_FILE}: {e}")
        sys.exit(1)


# Step 1: Download YouTube video as WAV audio
try:
    if (should_execute('download', CONTINUE_FROM)):
        DROPBOX_LINK, AUDIO_FILE_PATH = download_youtube_as_wav(YOUTUBE_LINK, AUDIO_FILE_DIR)
        if not AUDIO_FILE_PATH:
            raise Exception("AUDIO_FILE_PATH must be defined when continuing from a later step.")
        else:
            # create a json file to store youtube link, audio file path and dropbox link
            params = {
                'audio_file_path': str(AUDIO_FILE_PATH),
                'dropbox_link': DROPBOX_LINK
            }
            with open(BASE_DIR / "jsons" / f"{AUDIO_FILE_PATH.stem}.json", 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
except Exception as e:
    print(f"Error during download step: {e}")
    print(traceback.format_exc())
    sys.exit(1)

if not AUDIO_FILE_PATH:
    print("ERROR: AUDIO_FILE_PATH is not defined. Either Start from 'download' step or provide --i input json file.")
    sys.exit(1)
    
# Step 2: Update the other paths based on the downloaded audio file
OUTPUT_DIR = BASE_DIR / "output" / AUDIO_FILE_PATH.stem
TRANSCRIBE_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_transcribed_output.pkl"
PUNCTUATED_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_punctuated_output.txt"
SENTENCE_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_sentences_output.txt"
ALIGNED_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_aligned_output.pkl"
TRANSLITERATION_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_transliteration_output.pkl"
TRANSLATION_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_translation_output.pkl"
OUTPUT_CSV_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_results.csv"

# Audio segmentation based on silence
if (should_execute('segment', CONTINUE_FROM)):
    exported_chunk_paths = segment_dialogue(
        audio_file_path=AUDIO_FILE_PATH,
        output_dir=OUTPUT_DIR
    )

    if(exported_chunk_paths):
        print(f"Dialogue segments exported to: {OUTPUT_DIR}")
    else:
        print("No dialogue segments found.")
        exit(1)


# Transcribe audio-to-text (whisper)
# if (should_execute('transcribe', CONTINUE_FROM)):
#     transcribed_text = whisper_transcribe_chunks(
#         input_dir=OUTPUT_DIR,
#         output_file=TRANSCRIBE_OUTPUT_FILE
#     )

# TODO: Have a standard lang_code variable across the pipeline
# Transcrib1e audio-to-text (indic)
if (should_execute('transcribe', CONTINUE_FROM)):
    try:
        transcribed_text = indic_transcribe_chunks(
            lang_code='mr',
            exported_chunk_paths=exported_chunk_paths,
            output_file=TRANSCRIBE_OUTPUT_FILE
        )
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        sys.exit(1)

# Punctuation restoration
if (should_execute('punctuate', CONTINUE_FROM)):
    try:
        transcribed_text = {}
        with TRANSCRIBE_OUTPUT_FILE.open('rb') as f_in:
            transcribed_text = pickle.load(f_in)

            punctuated_text = restore_punctuation(" ".join(transcript['text'] for transcript in transcribed_text))

            with PUNCTUATED_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
                f_out.write(punctuated_text)
            print(f"Punctuated text saved to: {PUNCTUATED_OUTPUT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {TRANSCRIBE_OUTPUT_FILE}")
    except Exception as e:
        print(f"An error occurred during punctuation restoration: {e}")
        sys.exit(1)

# Preprocessing
if (should_execute('preprocess', CONTINUE_FROM)):
    try:
        preprocessed_text = []
        with PUNCTUATED_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
            # Get first line
            punc_text = f_in.read()

            preprocessed_text = preprocess_text(punc_text, 'mar_Deva')

            with SENTENCE_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
                f_out.write("\n".join(preprocessed_text))
                
            print(f"Preprocessed sentences saved to: {SENTENCE_OUTPUT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {TRANSCRIBE_OUTPUT_FILE}")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        sys.exit(1)


# After preprocessing, Identify the the proper timestamp for each sentenc chunk and get the corresponding audio file.
if (should_execute('align', CONTINUE_FROM)):
    try:
        with SENTENCE_OUTPUT_FILE.open('r', encoding='utf-8') as f_in, TRANSCRIBE_OUTPUT_FILE.open('rb') as t_in:
            transcribed_text = pickle.load(t_in)
            preprocessed_text = [line.strip() for line in f_in if line.strip()] 
            aligned_result = align_sentences_to_timestamps(transcribed_text, preprocessed_text, AUDIO_FILE_PATH)

            with ALIGNED_OUTPUT_FILE.open('wb') as f_out:
                pickle.dump(aligned_result, f_out)
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during alignment: {e}")
        sys.exit(1)


# Transliteration
if (should_execute('transliterate', CONTINUE_FROM)):
    try:
        with ALIGNED_OUTPUT_FILE.open('rb') as f_in:
            aligned_result = pickle.load(f_in)

            transliterated_result = transliterate_indic_to_english(aligned_result, 'mr')

            with TRANSLITERATION_OUTPUT_FILE.open('wb') as f_out:
                pickle.dump(transliterated_result, f_out)
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during transliteration: {e}")
        sys.exit(1)


# Translation
if (should_execute('translate', CONTINUE_FROM)):
    try:
        with TRANSLITERATION_OUTPUT_FILE.open('rb') as f_in:
            aligned_result = pickle.load(f_in)
            BATCH_SIZE = 1
            translated_result = []

            for i in range(0, len(aligned_result), BATCH_SIZE):
                chunk = aligned_result[i:i+BATCH_SIZE]
                print(f"Translating batch {i//BATCH_SIZE + 1} containing {len(chunk)} sentences...")
                translated = translate_indic_to_english(chunk, 'mr')
                translated_result.extend(translated)

            # save result into file after translation
            with TRANSLATION_OUTPUT_FILE.open('wb') as f_out:
                pickle.dump(translated_result, f_out)
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        sys.exit(1)

# Metrics Calculation
if (should_execute('metrics', CONTINUE_FROM)):
    try:
        with TRANSLATION_OUTPUT_FILE.open('rb') as f_in:
            translated_result = pickle.load(f_in)

        data = pd.DataFrame(translated_result)

        features = get_features(data, concat=False)

        difficulty_model = joblib.load('models\\al_random_forest_model.pkl')

        diffs = difficulty_model.predict(features.values)

        # creating a column 'difficulty' in data
        data['difficulty'] = diffs

        data.index.name = 'id'

        data['original_audio_file'] = DROPBOX_LINK

        data.to_csv(OUTPUT_CSV_FILE)

        # Upload CSV file to Google Drive using rclone
        upload_files([str(OUTPUT_CSV_FILE)], os.getenv("DROPBOX_CSV_FOLDER_PATH"))
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during metrics calculation: {e}")
        sys.exit(1)


# TODO: Clean up intermediate files only after reaching the end successfully