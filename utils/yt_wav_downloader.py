import os
import yt_dlp
from utils.rclone_helper import upload_files
import shutil

import dotenv
import tempfile
from pathlib import Path
dotenv.load_dotenv()

def download_youtube_as_wav(youtube_url: str, audio_output_dir: Path) -> str:
    """
    Download a YouTube video and convert it to WAV format.
    
    Args:
        youtube_url: The URL of the YouTube video to download.
    
    Returns:
        The path to the downloaded WAV file.
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'restrictfilenames': True,   # optional but recommended on Windows
            'windowsfilenames': True,    # optional but recommended on Windows
            'quiet': False,
            'no_warnings': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info to get the title
            info = ydl.extract_info(youtube_url, download=True)
            original_media = ydl.prepare_filename(info)
            temp_wav_path = os.path.splitext(original_media)[0] + ".wav"

        if not os.path.exists(temp_wav_path):
            candidates = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith('.wav')]
            if candidates:
                temp_wav_path = max(candidates, key=os.path.getmtime)
            else:
                raise FileNotFoundError(f"WAV not found in temp dir: {temp_dir}")

        title = info.get('title', 'audio')
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip() or "audio"
        permanent_wav_path = audio_output_dir / f"{clean_title}.wav"
        os.makedirs(audio_output_dir, exist_ok=True)
        shutil.move(temp_wav_path, permanent_wav_path)

        # Upload the downloaded audio file
        links = upload_files([permanent_wav_path], dropbox_folder=os.getenv("DROPBOX_AUDIO_FOLDER_PATH", ""))

   
    return links[0], permanent_wav_path


if __name__ == "__main__":
    # Example usage
    url = input("Enter YouTube URL: ")
    output_file = download_youtube_as_wav(url)
    print(f"Downloaded and uploaded WAV file: {output_file}")