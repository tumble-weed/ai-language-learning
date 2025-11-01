def whisper_transcribe_chunks(input_dir, output_file):
  print("Loading Whisper Model....")
  model = whisper.load_model("large")
  print("Whisper model loaded.")

  search_pattern = os.path.join(input_dir, '*.wav')
  chunk_files = sorted(glob.glob(search_pattern))

  if not chunk_files:
    print(f"No audio files found in {input_dir}.")
    return

  all_transcripts = []
  print(f"Found {len(chunk_files)} audio chunks to transcribe.")

  for i, chunk_file in enumerate(chunk_files):
    print(f"Transcribing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}...")

    result = model.transcribe(chunk_file, language='mr', fp16=False)
    transcribed_text = result['text'].strip()
    all_transcripts.append(transcribed_text)
    print(f"Transcription from {os.path.basename(chunk_file)}: {transcribed_text}")

  final_text = " ".join(all_transcripts)

  try:
    with open(output_file, 'w', encoding='utf-8') as f:
      f.write(final_text)
    print(f"Transcriptions saved to {output_file}")
  except Exception as e:
    print(f"Error saving transcriptions: {e}")

  return all_transcripts
