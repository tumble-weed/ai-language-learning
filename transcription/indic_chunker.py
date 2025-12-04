from transformers import AutoModel, AutoConfig
import torch, torchaudio
import os
import pickle

HF_MODELS_DIR = ".\\huggingface_models"
HF_TOKEN = os.getenv("HF_TOKEN")

model_name = "ai4bharat/indic-conformer-600m-multilingual"

# torchaudio.set_audio_backend("soundfile")

config = AutoConfig.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    model_name,
    config=config,
    use_auth_token=HF_TOKEN,
    trust_remote_code=True,
)

def indic_transcribe_chunks(lang_code, exported_chunk_paths, output_file=None):
  all_transcripts = []

  for i, file_info in enumerate(exported_chunk_paths):
    print(f"Transcribing chunk {i+1}/{len(exported_chunk_paths)}: {os.path.basename(file_info[0])}...")

    # Load an audio file
    wav, sr = torchaudio.load(file_info[0])
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
      all_transcripts.append({"filepath": file_info[0], "text": transcription_rnnt, "start": file_info[1], "end": file_info[2]})

    print(f"RNNT Transcription from {os.path.basename(file_info[0])}: {transcription_rnnt}")


  # Storing all_transcripts all dictionaries to output file
  if output_file:
     with open(output_file, 'wb') as f_out:
        pickle.dump(all_transcripts, f_out)
        
  return all_transcripts

