from transformers import AutoModel
import torch, torchaudio
import os
import glob
from pydub import AudioSegment

HF_MODELS_DIR = ".\\huggingface_models"
HF_TOKEN = os.getenv("HF_TOKEN")

torchaudio.set_audio_backend("soundfile")

model = AutoModel.from_pretrained(
  "ai4bharat/indic-conformer-600m-multilingual", 
  trust_remote_code=True,
  cache_dir=HF_MODELS_DIR,
  token=HF_TOKEN
)

def indic_transcribe_chunks(input_dir, lang_code, output_file=None):
  
  search_pattern = os.path.join(input_dir, '*.wav')
  chunk_files = sorted(glob.glob(search_pattern))

  if not chunk_files:
    print(f"No audio files found in {input_dir}.")
    return

  all_transcripts = {}
  print(f"Found {len(chunk_files)} audio chunks to transcribe.")

  for i, chunk_file in enumerate(chunk_files):
    print(f"Transcribing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}...")

    # Load an audio file
    wav, sr = torchaudio.load(chunk_file)
    wav = torch.mean(wav, dim=0, keepdim=True)

    target_sample_rate = 16000  # Expected sample rate
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)

    # # Perform ASR with CTC decoding
    # transcription_ctc = model(wav, lang_code, "ctc")
    # print("CTC Transcription:", transcription_ctc)

    # Perform ASR with RNNT decoding
    transcription_rnnt = model(wav, lang_code, "rnnt")
    if transcription_rnnt != "":
      all_transcripts[chunk_file.split('\\')[-1]] = transcription_rnnt

    print(f"RNNT Transcription from {os.path.basename(chunk_file)}: {transcription_rnnt}")


  if output_file:
    # final_text = "\n".join([f"{k}<|transcription|>{v}" for k, v in all_transcripts.items()])
    final_text = " ".join(all_transcripts.values())
    try:
      with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
      print(f"Transcriptions saved to {output_file}")
    except Exception as e:
      print(f"Error saving transcriptions: {e}")

  return all_transcripts

