import os
# from segmentation.webrtc_dialogue_segmentation import segment_dialogue
# from transcription.whisper_chunker import whisper_transcribe_chunks
# from transcription.indic_chunker import indic_transcribe_chunks
# from restore_punctuation.indic_punc_resto import restore_punctuation
# from preprocessing.indic_preprocessor import preprocess_text
from sentence_alignment.align_sentences import align_sentences_to_timestamps
from metrics.hindi_models import IndicReadabilityRH1, IndicReadabilityRH2
from metrics.wfr import WordFrequencyMetric
from metrics.sl import SentenceLengthMetric
import csv
from pathlib import Path
import dotenv
import pickle

dotenv.load_dotenv()  # Load environment variables from .env file


BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE_PATH = BASE_DIR / "input" / "test1.wav"
OUTPUT_DIR = BASE_DIR / "output" / "test1_dialogues"
TRANSCRIBE_OUTPUT_FILE = BASE_DIR / "output" / "test1_transcribed_output.pkl"
PUNCTUATED_OUTPUT_FILE = BASE_DIR / "output" / "test1_punctuated_output.txt"
SENTENCE_OUTPUT_FILE = BASE_DIR / "output" / "test1_sentences_output.txt"
OUTPUT_CSV_FILE = BASE_DIR / "output" / "test1_results.csv"

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
#     lang_code='te',
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

#         preprocessed_text = preprocess_text(punc_text, 'tel_Telu')

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

with SENTENCE_OUTPUT_FILE.open('r', encoding='utf-8') as f_in, TRANSCRIBE_OUTPUT_FILE.open('rb') as t_in:
    transcribed_text = pickle.load(t_in)
    preprocessed_text = [line.strip() for line in f_in if line.strip()] 
    aligned_result = align_sentences_to_timestamps(transcribed_text, preprocessed_text, AUDIO_FILE_PATH)

# Calculate difficulty score
rh1 = IndicReadabilityRH1()
rh2 = IndicReadabilityRH2()
# wrf = WordFrequencyMetric()
sl = SentenceLengthMetric()

# frequencies = wrf.get_frequencies()

results_list = []
for t in aligned_result:
    rh1Res = rh1.compute(t["sentence"])
    rh2Res = rh2.compute(t["sentence"])
    avg = (rh1Res + rh2Res) / 2
    slRes = sl.compute(t["sentence"])
    # wfrRes = wrf.compute(t, frequencies)

    row_data = {
        "Audio_file": t["audio_file"],
        "Text": t["sentence"],
        "RH1_Result": rh1Res,
        "RH2_Result": rh2Res,
        "RH_Average": avg,
        "SL_Result": slRes
        # "WFR_Result": wfrRes
    }
    results_list.append(row_data)

sorted_results = sorted(
    results_list, 
    key=lambda item: (item['SL_Result'], item['RH_Average']),
    reverse=False 
)



# Can also export to .json if needed. Exporting to CSV for simplicity.
header = ["Audio_file", "Text", "RH1_Result", "RH2_Result", "RH_Average", "SL_Result"]
try:
    with OUTPUT_CSV_FILE.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"\nProcessing complete. Sorted results saved to {OUTPUT_CSV_FILE}")
except Exception as e:
    print(f"An error occurred: {e}")


# Create a frontend to visualize the result better