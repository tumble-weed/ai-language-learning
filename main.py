import os
from utils.yt_wav_downloader import download_youtube_as_wav
# from segmentation.webrtc_dialogue_segmentation import segment_dialogue
# from transcription.whisper_chunker import whisper_transcribe_chunks
# from transcription.indic_chunker import indic_transcribe_chunks
# from restore_punctuation.indic_punc_resto import restore_punctuation
# from preprocessing.indic_preprocessor import preprocess_text
# from sentence_alignment.align_sentences import align_sentences_to_timestamps
# from transliteration.indic_en_trlit import transliterate_indic_to_english
# from translation.indic_en_translation import translate_indic_to_english
from utils.rclone_helper import upload_files
# from metrics.english_metrics import get_features
# from metrics.hindi_models import IndicReadabilityRH1, IndicReadabilityRH2
# from metrics.wfr import WordFrequencyMetric
# from metrics.sl import SentenceLengthMetric
# import csv
from pathlib import Path
import dotenv
import pickle
import pandas as pd
import joblib

dotenv.load_dotenv()  # Load environment variables from .env file


BASE_DIR = Path(__file__).resolve().parent
YOUTUBE_LINK = "https://www.youtube.com/watch?v=HY6DU2ABkSk" 
AUDIO_FILE_DIR = BASE_DIR / "input"

# Step 1: Download YouTube video as WAV audio
link, AUDIO_FILE_PATH = download_youtube_as_wav(YOUTUBE_LINK, AUDIO_FILE_DIR)

# Step 2: Update the other paths based on the downloaded audio file
OUTPUT_DIR = BASE_DIR / "output" / AUDIO_FILE_PATH.stem
TRANSCRIBE_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_transcribed_output.pkl"
PUNCTUATED_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_punctuated_output.txt"
SENTENCE_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_sentences_output.txt"
TRANSLITERATION_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_transliteration_output.pkl"
TRANSLATION_OUTPUT_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_translation_output.pkl"
OUTPUT_CSV_FILE = BASE_DIR / "output" / f"{AUDIO_FILE_PATH.stem}_results.csv"

# links = upload_files([str(AUDIO_FILE_PATH)], os.getenv("DROPBOX_AUDIO_FOLDER_PATH"))

# Audio segmentation based on silence
# exported_chunk_paths = segment_dialogue(
#     audio_file_path=AUDIO_FILE_PATH,
#     output_dir=OUTPUT_DIR
# )


# if(exported_chunk_paths):
#   print(f"Dialogue segments exported to: {OUTPUT_DIR}")
# else:
#   print("No dialogue segments found.")
#   exit(1)

# Transcribe audio-to-text

# transcribed_text = whisper_transcribe_chunks(
#     input_dir=OUTPUT_DIR,
#     output_file=TRANSCRIBE_OUTPUT_FILE
# )


# transcribed_text = indic_transcribe_chunks(
#     lang_code='mr',
#     exported_chunk_paths=exported_chunk_paths,
#     output_file=TRANSCRIBE_OUTPUT_FILE
# )

# punctuated_text = restore_punctuation(" ".join(transcript['text'] for transcript in transcribed_text))

# try:
#     transcribed_text = {}
#     with TRANSCRIBE_OUTPUT_FILE.open('rb') as f_in:
#         transcribed_text = pickle.load(f_in)

#         punctuated_text = restore_punctuation(" ".join(transcript['text'] for transcript in transcribed_text))

#         with PUNCTUATED_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
#             f_out.write(punctuated_text)
#         print(f"Punctuated text saved to: {PUNCTUATED_OUTPUT_FILE}")
# except FileNotFoundError:
#     print(f"ERROR: Input file not found at {TRANSCRIBE_OUTPUT_FILE}")

# Preprocessing
# preprocessed_text = preprocess_text(punctuated_text, 'mar_Deva')


# try:
#     preprocessed_text = []
#     with PUNCTUATED_OUTPUT_FILE.open('r', encoding='utf-8') as f_in:
#         # Get first line
#         punc_text = f_in.read()

#         preprocessed_text = preprocess_text(punc_text, 'mar_Deva')

#         with SENTENCE_OUTPUT_FILE.open('w', encoding='utf-8') as f_out:
#             f_out.write("\n".join(preprocessed_text))
            
#         print(f"Preprocessed sentences saved to: {SENTENCE_OUTPUT_FILE}")
# except FileNotFoundError:
#     print(f"ERROR: Input file not found at {TRANSCRIBE_OUTPUT_FILE}")
# except Exception as e:
#     print(f"An error occurred: {e}")

# After preprocessing, Identify the the proper timestamp for each sentenc chunk and get the 
# corresponding audio file

# aligned_result = align_sentences_to_timestamps(transcribed_text, preprocessed_text, AUDIO_FILE_PATH)
# aligned_result = transliterate_indic_to_english(aligned_result, 'te')

# with SENTENCE_OUTPUT_FILE.open('r', encoding='utf-8') as f_in, TRANSCRIBE_OUTPUT_FILE.open('rb') as t_in:
#     transcribed_text = pickle.load(t_in)
#     preprocessed_text = [line.strip() for line in f_in if line.strip()] 
#     aligned_result = align_sentences_to_timestamps(transcribed_text, preprocessed_text, AUDIO_FILE_PATH)
#     aligned_result = transliterate_indic_to_english(aligned_result, 'mr')

#     # save result into file after transliteration
#     with TRANSLITERATION_OUTPUT_FILE.open('wb') as f_out:
#         pickle.dump(aligned_result, f_out)


# BATCH_SIZE = 1
# translated_result = []
# for i in range(0, len(aligned_result), BATCH_SIZE):
#     chunk = aligned_result[i:i+BATCH_SIZE]
#     translated = translate_indic_to_english(chunk, 'mr')
#     translated_result.extend(translated)

# with TRANSLITERATION_OUTPUT_FILE.open('rb') as f_in:
#     aligned_result = pickle.load(f_in)
#     BATCH_SIZE = 1
#     translated_result = []

#     for i in range(0, len(aligned_result), BATCH_SIZE):
#         chunk = aligned_result[i:i+BATCH_SIZE]
#         print(f"Translating batch {i//BATCH_SIZE + 1} containing {len(chunk)} sentences...")
#         translated = translate_indic_to_english(chunk, 'mr')
#         translated_result.extend(translated)

#     # save result into file after translation
#     with TRANSLATION_OUTPUT_FILE.open('wb') as f_out:
#         pickle.dump(translated_result, f_out)


with TRANSLATION_OUTPUT_FILE.open('rb') as f_in:
    translated_result = pickle.load(f_in)

data = pd.DataFrame(translated_result)

features = get_features(data, concat=False)

difficulty_model = joblib.load('models\\al_random_forest_model.pkl')

diffs = difficulty_model.predict(features.values)

# creating a column 'difficulty' in data
data['difficulty'] = diffs

data.index.name = 'id'

data['original_audio_file'] = link

data.to_csv(OUTPUT_CSV_FILE)

# Upload CSV file to Google Drive using rclone
upload_files([str(OUTPUT_CSV_FILE)], os.getenv("DROPBOX_CSV_FOLDER_PATH"))