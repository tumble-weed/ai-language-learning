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
import yaml
import pandas as pd
import joblib
import json
import traceback
sys.stdout.reconfigure(encoding='utf-8')

dotenv.load_dotenv()  # Load environment variables from .env file


BASE_DIR = Path(__file__).resolve().parent

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

# TODO: Complete this map for indian languages
# How many languages are we targeting? 
lang_map = {
    'assamese': 'as',
    'bengali': 'bn',
    'bodo': 'brx',
    'dogri': 'doi',
    'gujarati': 'gu',
    'hindi': 'hi',
    'kannada': 'kn',
    'kashmiri': 'ks',
    'konkani': 'kok',
    'maithili': 'mai',
    'malayalam': 'ml',
    'manipuri': 'mni',
    'marathi': 'mr',
    'nepali': 'ne',
    'oriya': 'or',
    'punjabi': 'pa',
    'sanskrit': 'sa',
    'santali': 'sat',
    'sindhi': 'sd',
    'tamil': 'ta',
    'telugu': 'te',
    'urdu': 'ur',
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Language Learning Pipeline.")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        required=True
    )

    return parser.parse_args()

def get_step_index(step_name):
    return STEPS.index(step_name)

def should_execute(current_step, continue_from, continue_till):
    return get_step_index(current_step) >= get_step_index(continue_from) and get_step_index(current_step) <= get_step_index(continue_till)


args = parse_arguments()

with open(args.config, 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)

OUTPUT_DIR_NAME = Path(config.get("output-dir-name"))
CONTINUE_FROM = config.get("continue-from", "download")
CONTINUE_TILL = config.get("continue-till", "metrics")
YOUTUBE_LINK = config.get("yt-link")
LANGUAGE = config.get("lang", "marathi").lower()

if get_step_index(CONTINUE_FROM) > get_step_index(CONTINUE_TILL):
    print("ERROR: --continue-from step cannot be after --continue-till step.")
    sys.exit(1)

if not YOUTUBE_LINK:
    print("ERROR: YouTube link must be provided via --yt-link argument.")
    sys.exit(1)
    

# Create output directory if it doesn't exist
OUTPUT_DIR = BASE_DIR / "output" / OUTPUT_DIR_NAME
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define paths for intermediate and final outputs
DOWNLOAD_OUTPUT_FILE = OUTPUT_DIR / "download_output.json"
SEGMENT_OUTPUT_FILE = OUTPUT_DIR / "segments.json"
TRANSCRIBE_OUTPUT_FILE = OUTPUT_DIR / "transcription.json"
PUNCTUATED_OUTPUT_FILE = OUTPUT_DIR / "punctuation.txt"
PREPROCESS_OUTPUT_FILE = OUTPUT_DIR / "preprocess.txt"
ALIGNED_OUTPUT_FILE = OUTPUT_DIR / "align.json"
TRANSLITERATION_OUTPUT_FILE = OUTPUT_DIR / "transliteration.json"
TRANSLATION_OUTPUT_FILE = OUTPUT_DIR / "translation.json"
OUTPUT_CSV_FILE = OUTPUT_DIR / "results.csv"

# Step 1: Download YouTube video as WAV audio
try:
    if (should_execute('download', CONTINUE_FROM, CONTINUE_TILL)):
        DROPBOX_LINK, AUDIO_FILE_PATH = download_youtube_as_wav(YOUTUBE_LINK, OUTPUT_DIR)
        if not AUDIO_FILE_PATH:
            raise Exception("AUDIO_FILE_PATH must be defined when continuing from a later step.")
        else:
            # Save the dropbox link and audio file path in a json file for future reference
            with open(DOWNLOAD_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "dropbox_link": DROPBOX_LINK,
                    "audio_file_path": str(AUDIO_FILE_PATH)
                }, f, ensure_ascii=False, indent=4)
except Exception as e:
    print(f"Error during download step: {e}")
    print(traceback.format_exc())
    sys.exit(1)


# Always load AUDIO_FILE_PATH and DROPBOX_LINK from the download output file if it exists, regardless of the current step. This ensures that subsequent steps have access to these variables even if we are continuing from a later step.
if DOWNLOAD_OUTPUT_FILE.exists():
    with open(DOWNLOAD_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        download_info = json.load(f)
        DROPBOX_LINK = download_info.get("dropbox_link")
        AUDIO_FILE_PATH = Path(download_info.get("audio_file_path"))
else:
    print(f"ERROR: Download output file not found at {DOWNLOAD_OUTPUT_FILE}. Please ensure the download step has been executed successfully at least once.")
    sys.exit(1)


# Audio segmentation based on silence
if (should_execute('segment', CONTINUE_FROM, CONTINUE_TILL)):
    exported_chunks_info = segment_dialogue(
        audio_file_path=AUDIO_FILE_PATH,
    )

    if(exported_chunks_info):
        print(f"Dialogue segments exported successfully. Total segments: {len(exported_chunks_info)}")
        # Save the exported chunk paths in a json file for future reference
        with open(SEGMENT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # The exported chunk infor is a list of tuples containing (chunk_path, start_time, end_time). Saving this in json.
            json.dump(exported_chunks_info, f, ensure_ascii=False, indent=4)
    else:
        print("No dialogue segments found.")
        exit(1)


# Transcribe audio-to-text (whisper)
# if (should_execute('transcribe', CONTINUE_FROM, CONTINUE_TILL)):
#     transcribed_text = whisper_transcribe_chunks(
#         input_dir=OUTPUT_DIR,
#         output_file=TRANSCRIBE_OUTPUT_FILE
#     )

# Transcrib1e audio-to-text (indic)
if (should_execute('transcribe', CONTINUE_FROM, CONTINUE_TILL)):
    try:

        with open(SEGMENT_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            exported_chunks_info = json.load(f)

        transcribed_text = indic_transcribe_chunks(
            lang_code=lang_map.get(LANGUAGE, 'mr'),  # Default to Marathi if language not found
            exported_chunk_paths=exported_chunks_info,
        )

        with open(TRANSCRIBE_OUTPUT_FILE, "w", encoding="utf-8") as f:
            # save the list of dictionaries as a JSON array
            json.dump(transcribed_text, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        sys.exit(1)

# Punctuation restoration
if (should_execute('punctuate', CONTINUE_FROM, CONTINUE_TILL)):
    try:
        with TRANSCRIBE_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
            transcribed_text = json.load(f_in)

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
if (should_execute('preprocess', CONTINUE_FROM, CONTINUE_TILL)):
    try:
        preprocessed_text = []
        with PUNCTUATED_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
            # Get first line
            punc_text = f_in.read()

        preprocessed_text = preprocess_text(punc_text, lang_map.get(LANGUAGE, 'mr'))

        with PREPROCESS_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
            f_out.write("\n".join(preprocessed_text))
                
            print(f"Preprocessed sentences saved to: {PREPROCESS_OUTPUT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {PUNCTUATED_OUTPUT_FILE}")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        sys.exit(1)


# After preprocessing, Identify the the proper timestamp for each sentenc chunk and get the corresponding audio file.
if (should_execute('align', CONTINUE_FROM, CONTINUE_TILL)):
    try:
        with PREPROCESS_OUTPUT_FILE.open('r', encoding='utf-8') as f_in, TRANSCRIBE_OUTPUT_FILE.open('r', encoding='utf-8') as t_in:
            transcribed_text = json.load(t_in)
            preprocessed_text = [line.strip() for line in f_in if line.strip()] 

        aligned_result = align_sentences_to_timestamps(transcribed_text, preprocessed_text, AUDIO_FILE_PATH)

        with ALIGNED_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
            json.dump(aligned_result, f_out, ensure_ascii=False, indent=4)
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during alignment: {e}")
        sys.exit(1)


# Transliteration
if (should_execute('transliterate', CONTINUE_FROM, CONTINUE_TILL)):
    try:
        with ALIGNED_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
            aligned_result = json.load(f_in)

        transliterated_result = transliterate_indic_to_english(aligned_result, lang_map.get(LANGUAGE, 'mr'))

        with TRANSLITERATION_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
            json.dump(transliterated_result, f_out, ensure_ascii=False, indent=4)
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during transliteration: {e}")
        sys.exit(1)


# Translation
if (should_execute('translate', CONTINUE_FROM, CONTINUE_TILL)):
    try:
        with TRANSLITERATION_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
            aligned_result = json.load(f_in)

        BATCH_SIZE = 1
        translated_result = []

        for i in range(0, len(aligned_result), BATCH_SIZE):
            chunk = aligned_result[i:i+BATCH_SIZE]
            print(f"Translating batch {i//BATCH_SIZE + 1} containing {len(chunk)} sentences...")
            translated = translate_indic_to_english(chunk, lang_map.get(LANGUAGE, 'mr'))
            translated_result.extend(translated)

        # save result into file after translation
        with TRANSLATION_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
            json.dump(translated_result, f_out, ensure_ascii=False, indent=4)
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        sys.exit(1)

# Metrics Calculation
if (should_execute('metrics', CONTINUE_FROM, CONTINUE_TILL)):
    try:
        with TRANSLATION_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
            translated_result = json.load(f_in)

        data = pd.DataFrame(translated_result)

        features = get_features(data, concat=False)

        difficulty_model = joblib.load(BASE_DIR / "models" / "al_random_forest_model.pkl")

        diffs = difficulty_model.predict(features.values)

        # creating a column 'difficulty' in data
        data['difficulty'] = diffs

        data.index.name = 'id'

        data['original_audio_file'] = DROPBOX_LINK

        data.to_csv(OUTPUT_CSV_FILE)

        # Upload CSV file to Google Drive using rclone
        upload_files([str(AUDIO_FILE_PATH.stem) + str(OUTPUT_CSV_FILE)], os.getenv("DROPBOX_CSV_FOLDER_PATH"))
    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")
    except Exception as e:
        print(f"An error occurred during metrics calculation: {e}")
        sys.exit(1)



# TODO: Clean up intermediate files only after reaching the end successfully. 