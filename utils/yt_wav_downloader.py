import os
import yt_dlp
from rclone_helper import upload_files

import dotenv
import tempfile
dotenv.load_dotenv()

def download_youtube_as_wav(youtube_url: str) -> str:
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
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info to get the title
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get('title', 'audio')
            # Clean the title for filename
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            wav_path = os.path.join(temp_dir, f"{clean_title}.wav")
        
        # Upload the downloaded audio file
        links = upload_files([wav_path], dropbox_folder=os.getenv("DROPBOX_AUDIO_FOLDER_PATH", ""))
        
    return links[0]


if __name__ == "__main__":
    # Example usage
    url = input("Enter YouTube URL: ")
    output_file = download_youtube_as_wav(url)
    print(f"Downloaded and uploaded WAV file: {output_file}")