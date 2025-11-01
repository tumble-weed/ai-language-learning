from pathlib import Path
import webrtcvad
from pydub import AudioSegment

def segment_dialogue(audio_file_path, output_dir, vad_aggressiveness=3, min_speech_len_ms=250, min_silence_after_speech_ms=500, padding_ms=200):
    """Segment dialogue from an audio file.

    Accepts either strings or pathlib.Path for audio_file_path and output_dir.
    Returns a list of exported chunk file paths (strings).
    """
    audio_file_path = Path(audio_file_path)
    output_dir = Path(output_dir)

    if not audio_file_path.exists():
        print(f"Audio file {audio_file_path} not found")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        print(f"Loading audio file {audio_file_path}")
        sound = AudioSegment.from_file(str(audio_file_path))
        print(f"Audio file loaded successfully.")
    except Exception as e:
        # Common failure: missing ffmpeg/ffprobe in PATH — give a helpful hint
        err_str = str(e)
        print(f"Error loading audio file: {err_str}")
        if "ffprobe" in err_str or "ffmpeg" in err_str or isinstance(e, FileNotFoundError) or "No such file" in err_str:
            print("Hint: pydub requires ffmpeg/ffprobe (or avconv/avprobe). Make sure ffmpeg is installed and available on your PATH.")
            print(r"On Windows you can install ffmpeg and add its 'bin' folder to PATH, or set pydub.AudioSegment.converter and .ffprobe to point to the executables.")
        return []

    print("Preprocessing audio for VAD")
    vad_audio = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    print("Performing voice activity detection...")
    vad = webrtcvad.Vad(vad_aggressiveness)

    frame_duration_ms = 30
    frame_size_bytes = int(vad_audio.frame_rate * (frame_duration_ms / 1000.0) * vad_audio.sample_width)

    silence_frames_count = 0
    is_speech = False
    speech_segments = []
    start_time = 0

    for i in range(0, len(vad_audio.raw_data), frame_size_bytes):
        frame = vad_audio.raw_data[i:i+frame_size_bytes]
        if len(frame) < frame_size_bytes:
            break

        current_time_ms = (i / frame_size_bytes) * frame_duration_ms

        if vad.is_speech(frame, vad_audio.frame_rate):
            if not is_speech:
                start_time = current_time_ms
                is_speech = True
            silence_frames_count = 0
        else:
            if is_speech:
                silence_frames_count += 1
                if (silence_frames_count * frame_duration_ms > min_silence_after_speech_ms):
                    end_time = current_time_ms - min_silence_after_speech_ms
                    if (end_time - start_time) >= min_speech_len_ms:
                        speech_segments.append({'start': start_time, 'end': end_time})
                    is_speech = False

    if is_speech and (len(vad_audio) - start_time) >= min_speech_len_ms:
        speech_segments.append({'start': start_time, 'end': len(vad_audio)})

    if not speech_segments:
        print("No speech segments found.")
        return []

    print("Exporting dialogue segments...")
    exported_files = []
    for i, segment in enumerate(speech_segments):
        start_ms = segment['start'] - padding_ms
        end_ms = segment['end'] + padding_ms

        start_ms = max(0, start_ms)
        end_ms = min(len(sound), end_ms)

        chunk = sound[start_ms:end_ms]

        output_filename = f"chunk_{i:04d}.wav"
        output_path = output_dir / output_filename

        print(f"Exporting {output_filename}")
        # pydub export expects a filename (string)
        chunk.export(str(output_path), format="wav")
        exported_files.append(str(output_path))

    print("Dialogue segmentation completed.")
    return exported_files