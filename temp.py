# import streamlit as st
# import gdown
# from pathlib import Path
# import shutil

# st.title("Audio Snippet Player")

# row1 = {
#     "start_time": 2.0,
#     "end_time": 4.0
# }

# row2 = {
#     "start_time": 5.0,
#     "end_time": 8.0
# }

# def render_audio_player(row):
#     safe_id = f"audio_player_{row['start_time']}_{row['end_time']}".replace('.', '_')
#     audio_html = f"""
#     <audio id="{safe_id}" controls name="media">
#         <source src="https://www.dropbox.com/scl/fi/7q8usuctj1jrpmqo6zfuk/test5.wav?rlkey=dd8u2pzjyu8ext350zh3wfxlk&st=hivng2og&raw=1" type="audio/wav">
#     </audio>

#     <script>
#     const audio = document.getElementById("{safe_id}");
#     const start = {row['start_time']};
#     const end = {row['end_time']};

#     audio.onplay = function() {{
#         if (audio.currentTime < start || audio.currentTime > end) {{
#             audio.currentTime = start;
#         }}
#         const interval = setInterval(() => {{
#             if (audio.currentTime >= end) {{
#                 audio.pause();
#                 audio.currentTime = start;
#                 clearInterval(interval);
#             }}
#         }}, 100);
#     }};
#     </script>
#     """
#     st.markdown(audio_html, unsafe_allow_html=True)


# st.subheader("Snippet Player")
# render_audio_player(row1)
# # render_audio_player(row2)

import streamlit as st
from streamlit.components.v1 import html

# Load shared JavaScript only once
if "shared_js_loaded" not in st.session_state:
    st.session_state.shared_js_loaded = True

    html("""
    <script>
    window.snippetPlayer = function(audio, start, end) {

        audio.onplay = function () {

            // Restart at start if outside the range
            if (audio.currentTime < start || audio.currentTime > end) {
                audio.currentTime = start;
            }

            // Clear previous intervals
            if (audio._interval) {
                clearInterval(audio._interval);
            }

            // Enforce snippet end
            audio._interval = setInterval(() => {
                if (audio.currentTime >= end) {
                    audio.pause();
                    audio.currentTime = start;
                    clearInterval(audio._interval);
                }
            }, 80);
        };
    };
    </script>
    """, height=0)     # hidden component

def render_audio_player(start: float, end: float, audio_url: str):
    component_html = f"""
    <audio controls
           src="{audio_url}#t={start},{end}"
           onplay="window.snippetPlayer(this, {start}, {end})">
    </audio>
    """

    html(component_html, height=80)

AUDIO_LINK = "https://www.dropbox.com/scl/fi/7q8usuctj1jrpmqo6zfuk/test5.wav?rlkey=dd8u2pzjyu8ext350zh3wfxlk&st=hivng2og&raw=1"

st.write("### Snippet 1 (0 → 3 sec)")
render_audio_player(0, 3, AUDIO_LINK)

st.write("### Snippet 2 (3 → 6 sec)")
render_audio_player(3, 6, AUDIO_LINK)
