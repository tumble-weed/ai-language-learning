import asyncio
import logging
import warnings
import sys
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Suppress all aioice and asyncio warnings/errors
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aioice.ice").setLevel(logging.CRITICAL)
logging.getLogger("aioice.stun").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Custom exception handler to suppress event loop errors
def custom_exception_handler(loop, context):
    """Suppress specific asyncio errors that occur during WebRTC cleanup"""
    exception = context.get("exception")
    message = context.get("message", "")
    
    # List of error patterns to suppress
    suppress_patterns = [
        "NoneType",
        "is_alive",
        "sendto",
        "call_exception_handler",
        "_fatal_error",
        "Fatal write error on datagram transport"
    ]
    
    if exception:
        error_msg = str(exception)
        if any(pattern in error_msg for pattern in suppress_patterns):
            return
    
    if any(pattern in message for pattern in suppress_patterns):
        return
    
    # For other exceptions, log them but don't crash
    if exception:
        print(f"Asyncio exception: {exception}", file=sys.stderr)

# Set up event loop with custom exception handler
def setup_event_loop():
    """Initialize event loop with proper exception handling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.set_exception_handler(custom_exception_handler)
    return loop

# Initialize event loop
setup_event_loop()

# Ensure event loop policy is set for Windows compatibility
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# RTC Configuration with STUN and public TURN servers
RTC_CONFIGURATION = {
    "iceServers": [
        # Google STUN servers
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        # OpenRelay TURN servers (multiple for redundancy)
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        # Alternative TURN server
        {
            "urls": [
                "turn:relay1.expressturn.com:3478",
            ],
            "username": "efU4B4EU4K6X8XLM3B",
            "credential": "asFUl2sMxaHX01Rt",
        },
    ],
    "iceTransportPolicy": "all",  # Use both STUN and TURN
    "iceCandidatePoolSize": 10,  # Pre-gather ICE candidates
}


def main():
    st.title("🎤 WebRTC Audio Streaming")
    
    # Add instructions
    st.markdown("""
    ### Instructions:
    1. Click **START** below to begin audio streaming
    2. Allow microphone access when prompted by your browser
    3. Wait for the connection to establish (may take 10-30 seconds)
    
    **Note:** If connection fails:
    - Check your microphone permissions
    - Disable VPN if active
    - Try a different browser (Chrome/Edge recommended)
    - Check firewall settings
    """)
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="audio-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": False,
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                }
            },
            sendback_audio=False,
            async_processing=True,
        )
        
        # Display connection status
        if webrtc_ctx.state.playing:
            st.success("✅ Audio streaming is active!")
            st.info("🔴 Recording... Speak into your microphone")
        elif webrtc_ctx.state.signalling:
            st.warning("⏳ Connecting... Please wait (this may take up to 30 seconds)")
        else:
            st.info("👆 Click **START** above to begin streaming")
            
    except Exception as e:
        # Suppress event loop cleanup errors
        if "NoneType" not in str(e) and "is_alive" not in str(e):
            st.error(f"❌ Error: {e}")
            st.info("Try refreshing the page or checking your network connection")
            raise


if __name__ == "__main__":
    main()