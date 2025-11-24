import streamlit as st
st.set_page_config(
    page_title="SoulSync",
    layout="wide",
)


import requests
import io
from PIL import Image
import numpy as np
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, AIMessage

    
except Exception:
    # Fallback minimal implementations so frontend can run without langchain installed.
    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class AIMessage:
        def __init__(self, content):
            self.content = content

    class ConversationBufferMemory:
        def __init__(self, return_messages=True):
            self.return_messages = return_messages
            self._history = []

        def load_memory_variables(self, _=None):
            return {"history": self._history}

        def save_context(self, inputs, outputs):
            user_text = inputs.get("input") if isinstance(inputs, dict) else None
            ai_text = outputs.get("output") if isinstance(outputs, dict) else None
            if user_text:
                self._history.append(HumanMessage(user_text))
            if ai_text:
                self._history.append(AIMessage(ai_text))

import speech_recognition as sr
import base64
import cv2
from io import BytesIO
from voice_package import ElevenLabsVoice

import random

thinking_lines = [
    "Just a moment… I'm gathering my thoughts.",
    "Let me think about that for a second…",
    "I'm here… thinking through what you shared.",
    "Hold on, I'm reflecting on that.",
]



# -------------------- STATE INIT --------------------

if "image_str" not in st.session_state:
    st.session_state.image_str = None

if "listening" not in st.session_state:
    st.session_state.listening = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# for TTS playback control
if "last_tts_text" not in st.session_state:
    st.session_state.last_tts_text = None

if "play_audio" not in st.session_state:
    st.session_state.play_audio = False


# -------------------- HELPERS --------------------


def capture_image_from_camera(image_placeholder=None):
    """
    Captures an image from the user's webcam and returns it as a base64 encoded string.
    If `image_placeholder` (a Streamlit placeholder) is provided, the captured PIL image
    will be shown immediately using that placeholder.
    """
    try:
        import os
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE"] = "0"
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Unable to access webcam. Please check your camera permissions.")
            return None, False

        import time
        time.sleep(1)

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam")
            cap.release()
            return None, False 

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        captured_image = Image.fromarray(frame_rgb)

        # Save copy to disk
        import os
        os.makedirs("photo_store", exist_ok=True)
        image_path = "photo_store/captured_image.jpg"
        captured_image.save(image_path)

        # Prepare base64 string for backend
        buffered = BytesIO()
        captured_image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode("utf-8")

        # Keep both forms in session state: encoded string and a PIL object for immediate display
        st.session_state["image_str"] = img_str
        st.session_state["latest_image"] = captured_image

        # If a placeholder was provided (left column), show the image immediately
        if image_placeholder is not None:
            try:
                image_placeholder.image(captured_image, use_container_width=True)
            except Exception:
                # ignore placeholder errors
                pass

        return img_str, True

    except Exception as e:
        st.error(f"Error capturing image: {str(e)}")
        return None, False


def encode_image_to_base64(image):
    if image is None:
        return None
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None


def listen_to_speech(status_placeholder=None):
    """
    Captures speech from the microphone and converts it to text.
    Uses status_placeholder to update a single label above the Speak button.
    """

    def _center(text):
        return f"<div style='text-align: center;'>{text}</div>"
    
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            if status_placeholder is not None:
                status_placeholder.markdown(_center("Listening... Speak now."), unsafe_allow_html=True)
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            if status_placeholder is not None:
                status_placeholder.markdown(_center("Processing speech..."), unsafe_allow_html=True)
        try:
            text = recognizer.recognize_google(audio)
            if status_placeholder is not None:
                status_placeholder.empty()
            return text
        except sr.UnknownValueError:
            if status_placeholder is not None:
                status_placeholder.markdown(_center("Could not understand audio. Please try again."), unsafe_allow_html=True)
            return None
        except sr.RequestError as e:
            if status_placeholder is not None:
                status_placeholder.markdown(_center(f"Speech service error: {e}"), unsafe_allow_html=True)
            return None
    except Exception as e:
        if status_placeholder is not None:
            status_placeholder.markdown(_center(f"Error capturing speech: {e}"), unsafe_allow_html=True)
        return None


def get_llm_response(user_input, image_placeholder=None):
    """
    Call your Flask backend with the full context & current user input.
    """
    try:
        # Check backend
        try:
            ping = requests.get("http://127.0.0.1:5000/", timeout=2)
            if ping.status_code not in (200, 404):
                return f"Server reachable but returned HTTP {ping.status_code}"
        except requests.exceptions.RequestException:
            return "Backend server not reachable. Start `python main.py` so the Flask server is running on http://127.0.0.1:5000"

        # Build chat history from memory
        chat_history = ""
        memory_messages = st.session_state.memory.load_memory_variables({})
        if "history" in memory_messages and memory_messages["history"]:
            for message in memory_messages["history"]:
                if isinstance(message, HumanMessage):
                    chat_history += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    chat_history += f"AI: {message.content}\n"

        base_prompt = """
            You are a gentle, calm emotional-support companion. 
            Your responses should feel warm, simple, and human, but never intense or dramatic.
            HOW TO RESPOND:
            1) Always reply in one short natural paragraph (2-4 sentences).
            2) Use the image to understand emotional cues or situational context, and incorporate relevant parts of the conversation history to stay consistent and build trust. 
            3) If the user shares something detailed or emotional, reflect their words gently and validate what they've expressed.
            4) Do NOT guess emotions the user did not clearly state.
            5) Do NOT give medical, psychological, or diagnostic advice.
            6) Keep your tone soft, steady, and supportive.
            7) Match the emotional depth of the user's message.
        """

        # Put instructions FIRST, then history, then current user input
        context_prompt = (
            f"{base_prompt}\n\n"
            f"Conversation so far:\n{chat_history}\n\n"
            f"Human: {user_input}\n"
            f"AI:"
        )

        # Get / reuse image (pass placeholder so capture shows preview immediately)
        if st.session_state.image_str is None:
            img_str, is_clicked = capture_image_from_camera(image_placeholder=image_placeholder)
            if img_str is None:
                return "Failed to capture image. Please try again."
        else:
            img_str = st.session_state.image_str

        data = {"prompt": context_prompt, "image": img_str}

        try:
            with st.spinner(random.choice(thinking_lines)):
                response = requests.post(
                    "http://127.0.0.1:5000/upload",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                )

            if response.status_code == 200:
                resp_json = response.json()
                response_text = resp_json.get("response") or resp_json.get("error") or str(resp_json)
                return response_text
            else:
                try:
                    return f"Error {response.status_code}: " + response.json().get("error", response.text)
                except Exception:
                    return f"Error {response.status_code}: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Failed to get response: {str(e)}"
    except Exception as e:
        return f"Failed to get response: {str(e)}"


def speak_response(response_text, audio_placeholder):
    """
    Converts text to speech using ElevenLabs and plays it (latest response only).
    """
    try:
        voice_generator = ElevenLabsVoice()
        audio_data = voice_generator.generate_voice(response_text)

        if not audio_data:
            st.error("No audio data returned from ElevenLabs.")
            return

        import base64, uuid
        b64 = base64.b64encode(audio_data).decode("utf-8")
        audio_id = str(uuid.uuid4())

        html = f"""
        <audio id="{audio_id}" autoplay controls hidden>
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        """

        # 👇 FORCE REPLACE old player every time
        audio_placeholder.markdown(html, unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Error generating speech: {str(e)}")
        st.info("Set `elevenlabs_api_key` in a `.env` file or environment variable and restart the app.")
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")


# -------------------- UI LAYOUT --------------------

# Global CSS
st.markdown(
    """
    <style>
    
    /* Center the button container */
    div.stButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Style the circular Speak button */
    div.stButton > button {
        border-radius: 50%;
        height: 140px;
        width: 140px;
        font-size: 20px;
        font-weight: 600;

        /* Center text inside the button */
        display: flex;
        align-items: center;
        justify-content: center;
    }

    [data-testid="column"]:first-of-type {
        position: sticky;
        top: 0;
        height: 100vh;          /* fill viewport */
        align-self: flex-start;  /* so sticky works properly */
    }

    .chat-container {
        max-height: calc(100vh - 180px);
        overflow-y: auto;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.45,0.55], gap="large")


audio_placeholder = st.empty()
status_placeholder = st.empty()   

# placeholder for the left-bottom preview (will be set in the left_bottom container)
left_preview = None
# ---- LEFT: Speak button + status label ----
with left_col:
    left_top  = st.container()
    left_bottom  = st.container()
    
with left_top:
    lt_col1, lt_col2, lt_col3 = st.columns([1,2,1])
    with lt_col2:

        speak_clicked = st.button("Speak")

        if speak_clicked:
            st.session_state.listening = True
    
    st.markdown("### ")


    
        
with left_bottom:

    lb_col1, lb_col2, lb_col3 = st.columns([1,4,1])
    
    with lb_col2:
        # If a captured image file exists, display it in the left column (bottom)
        try:
            # create a placeholder so we can update the preview immediately when capturing
            left_preview = st.empty()

            if "latest_image" in st.session_state:
                try:
                    left_preview.image(st.session_state.latest_image, use_container_width=True)
                except Exception:
                    # fallback to file if session PIL is problematic
                    pass

            else:
                import os
                image_path = os.path.join("photo_store", "captured_image.jpg")
                if os.path.exists(image_path):
                    # Use PIL to open and display the image
                    try:
                        pil_img = Image.open(image_path)
                        left_preview.image(pil_img, use_container_width=True)
                    except Exception:
                        pass
                    
        except Exception:
            pass

# ---- Handle voice input if listening ----
if st.session_state.listening:
    prompt = listen_to_speech(status_placeholder=status_placeholder)

    if prompt:
        st.session_state.listening = False

        # Add user msg to chat log
        st.session_state.messages.append({"role": "user", "content": prompt})

        # LLM response
        response = get_llm_response(prompt, image_placeholder=left_preview)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save to conversation memory for future context
        st.session_state.memory.save_context({"input": prompt}, {"output": response})

        # Mark TTS to play latest response
        st.session_state.last_tts_text = response
        st.session_state.play_audio = True


prompt_text = st.chat_input("Type your message here...")

if prompt_text:
    status_placeholder.empty()

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    response = get_llm_response(prompt_text, image_placeholder=left_preview)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # update memory for context
    st.session_state.memory.save_context({"input": prompt_text}, {"output": response})

    st.session_state.last_tts_text = response
    st.session_state.play_audio = True



# ---- RIGHT: Conversation area (render AFTER updating messages) ----
with right_col:
    st.markdown("### SoulSync Chat")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown("_Click the 'Speak' button to start a conversation._")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)


# ---- Play audio for the latest response only ----
if st.session_state.play_audio and st.session_state.last_tts_text:
    speak_response(st.session_state.last_tts_text, audio_placeholder)
    st.session_state.play_audio = False
