import streamlit.components.v1 as components
import streamlit as st
import gdown
from pathlib import Path
import shutil

st.title("Audio Snippet Player")

row1 = {
    "start_time": 2.0,
    "end_time": 4.0
}

row2 = {
    "start_time": 5.0,
    "end_time": 8.0
}

def render_audio_player(row):
    safe_id = f"audio_player_{row['start_time']}_{row['end_time']}".replace('.', '_')
    audio_html = f"""
    <audio id="{safe_id}" controls name="media">
        <source src="https://www.dropbox.com/scl/fi/2wdfn12n6pyrn9uybe5ys/test5.wav?rlkey=98locxln1cc3n36hpghd6s72q&raw=1" type="audio/wav">
    </audio>

    <script>
    const audio = document.getElementById("{safe_id}");
    const start = {row['start_time']};
    const end = {row['end_time']};

    audio.onplay = function() {{
        if (audio.currentTime < start || audio.currentTime > end) {{
            audio.currentTime = start;
        }}
        const interval = setInterval(() => {{
            if (audio.currentTime >= end) {{
                audio.pause();
                audio.currentTime = start;
                clearInterval(interval);
            }}
        }}, 100);
    }};
    </script>
    """
    components.html(audio_html, height=100)


st.subheader("Snippet Player")
render_audio_player(row1)
render_audio_player(row2)





# import streamlit as st
# from streamlit.components.v1 import html

# # Load shared JavaScript only once
# if "shared_js_loaded" not in st.session_state:
#     st.session_state.shared_js_loaded = True

#     html("""
#     <script>
#     window.snippetPlayer = function(audio, start, end) {

#         audio.onplay = function () {

#             // Restart at start if outside the range
#             if (audio.currentTime < start || audio.currentTime > end) {
#                 audio.currentTime = start;
#             }

#             // Clear previous intervals
#             if (audio._interval) {
#                 clearInterval(audio._interval);
#             }

#             // Enforce snippet end
#             audio._interval = setInterval(() => {
#                 if (audio.currentTime >= end) {
#                     audio.pause();
#                     audio.currentTime = start;
#                     clearInterval(audio._interval);
#                 }
#             }, 80);
#         };
#     };
#     </script>
#     """, height=0)     # hidden component

# def render_audio_player(start: float, end: float, audio_url: str):
#     component_html = f"""
#     <audio controls
#            src="{audio_url}#t={start},{end}"
#            onplay="window.snippetPlayer(this, {start}, {end})">
#     </audio>
#     """

#     html(component_html, height=80)

# AUDIO_LINK = "https://www.dropbox.com/scl/fi/7q8usuctj1jrpmqo6zfuk/test5.wav?rlkey=dd8u2pzjyu8ext350zh3wfxlk&st=hivng2og&raw=1"

# st.write("### Snippet 1 (0 → 3 sec)")
# render_audio_player(0, 3, AUDIO_LINK)

# st.write("### Snippet 2 (3 → 6 sec)")
# render_audio_player(3, 6, AUDIO_LINK)



# Get drive file using client id and secret
# import os
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# import dotenv
# import json

# dotenv.load_dotenv()

# SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
# FOLDER_ID = os.getenv("GOOGLE_DRIVE_CSV_FOLDER_ID")


# def get_token_data():
#     """Reads the token JSON string from the environment."""
#     token_string = os.environ.get("TOKEN_INFO")
#     if not token_string:
#         if os.path.exists("token.json"):
#             # load data from token.json into token_string
#             with open("token.json", "r") as token_file:
#                 token_string = token_file.read()
#         else:
#             raise Exception("TOKEN_INFO not found in environment variables.")
#     return token_string

# def update_token_data(creds):
#     """Writes the updated token JSON string back to the environment."""
#     updated_token_string = creds.to_json()
    
#     os.environ["TOKEN_INFO"] = updated_token_string
#     print(f"✅ Successfully refreshed token and updated 'TOKEN_INFO'.")
#     return updated_token_string


# def get_drive_service():
#     if os.path.exists("token.json"):
#         return build("drive", "v3",
#         credentials=Credentials.from_authorized_user_file("token.json", SCOPES))

#     flow = InstalledAppFlow.from_client_secrets_file(
#         "client_secret.json", SCOPES
#     )

#     creds = flow.run_local_server(
#         host="127.0.0.1",
#         port=53682,
#         redirect_uri_trailing_slash=True
#     )

#     with open("token.json", "w") as token:
#         token.write(creds.to_json())

#     return build("drive", "v3", credentials=creds)

#     try:
#         token_data = get_token_data()
#         token_info = json.loads(token_data)

#         creds = Credentials.from_authorized_user_info(token_info, SCOPES)

#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         elif creds and creds.refresh_token:
#             if not creds.valid:
#                 creds.refresh(Request())
#         else:
#             raise Exception("No valid credentials available.")
        
#         update_token_data(creds)

#         return build("drive", "v3", credentials=creds)
#     except Exception as e:
#         print(f"⚠️ Error obtaining credentials: {e}")
#         return None



# def list_files(folder_id):
#     service = get_drive_service()

#     results = service.files().list(
#         q=f"'{folder_id}' in parents",
#         fields="files(id, name, mimeType)"
#     ).execute()

#     return results.get("files", [])

# # -------------------------------
# # USE IT
# # -------------------------------
# print(list_files(FOLDER_ID))


# from googleapiclient.discovery import build

# API_KEY = "AIzaSyB2h-O4vgIPrAPy_rR575wrv24DeX0z1Sw"
# FOLDER_ID = "1Qk1UrsJGFqW5FE3sopTw195xKueYNcVK"

# def list_public_folder_files(folder_id):
#     service = build("drive", "v3", developerKey=API_KEY)

#     results = service.files().list(
#         q=f"'{folder_id}' in parents and trashed = false",
#         fields="files(id, name, mimeType)"
#     ).execute()

#     return results.get("files", [])

# print(list_public_folder_files(FOLDER_ID))




