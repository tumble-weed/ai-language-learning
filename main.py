import os
# from segmentation.webrtc_dialogue_segmentation import segment_dialogue
# from transcription.whisper_chunker import whisper_transcribe_chunks
# from transcription.indic_chunker import indic_transcribe_chunks
from metrics.hindi_models import IndicReadabilityRH1, IndicReadabilityRH2
from metrics.wfr import WordFrequencyMetric
from metrics.sl import SentenceLengthMetric
import csv
from pathlib import Path
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file


BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE_PATH = BASE_DIR / "input" / "test2.mp3"
OUTPUT_DIR = BASE_DIR / "output" / "test2_dialogues"
TRANSCRIBE_OUTPUT_FILE = BASE_DIR / "output" / "test2_transcribed_output.txt"
OUTPUT_CSV_FILE = BASE_DIR / "output" / "test2_results.csv"

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
#     input_dir=OUTPUT_DIR,
#     output_file=TRANSCRIBE_OUTPUT_FILE
# )

# Calculate difficulty score
rh1 = IndicReadabilityRH1()
rh2 = IndicReadabilityRH2()
wrf = WordFrequencyMetric()
sl = SentenceLengthMetric()


try:
    with TRANSCRIBE_OUTPUT_FILE.open('r', encoding='utf-8') as f_in, \
         OUTPUT_CSV_FILE.open('w', encoding='utf-8', newline='') as f_out:

        # 1. Create a CSV writer object
        writer = csv.writer(f_out)

        # 2. Write the header row
        writer.writerow(["Chunk_ID", "Text", "RH1_Result", "RH2_Result", "RH_Average", "SL_Result", "WFR_Result"])

        # Preprocess each line to get sentences and chunk name separated by <|transcription|>
        chunk_transcriptions = {}
        for line in f_in:
            if "<|transcription|>" in line:
                chunk_name, transcription = line.split("<|transcription|>", 1)
                chunk_transcriptions[chunk_name.strip()] = transcription.strip()

        # Get word frequencies for all sentences
        frequencies = wrf.get_frequencies(chunk_transcriptions.values())

       

        # Loop through your input file
        for chunk_id, t in chunk_transcriptions.items():
            # Perform computations
            rh1Res = rh1.compute(t)
            rh2Res = rh2.compute(t)
            avg = (rh1Res + rh2Res) / 2
            slRes = sl.compute(t)
            wfrRes = wrf.compute(t, frequencies)

            # 3. Write the data as a row
            writer.writerow([chunk_id, t, rh1Res, rh2Res, avg, slRes, wfrRes])

    print(f"\nProcessing complete. Results saved to {OUTPUT_CSV_FILE}")

except FileNotFoundError:
    print(f"ERROR: Input file not found at {TRANSCRIBE_OUTPUT_FILE}")
except Exception as e:
    print(f"An error occurred: {e}")

"""
TODO:
Removing non-marathi sentences.
Processing incomplete sentences.

Active model learning models.
"""
