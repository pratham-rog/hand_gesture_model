import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


st.set_page_config(page_title="Virtual Keyboard & Air Mouse", page_icon="🖐️", layout="wide")

st.title("Virtual Keyboard & Air Mouse")
st.caption("Streamlit demo of the hand-tracking core behind the desktop app")

st.markdown(
    """
This Streamlit version is a deployable web demo. It can analyze a webcam snapshot or uploaded image,
detect hands with MediaPipe, and draw landmarks on top of the frame.

The original desktop app still provides OS-level mouse and keyboard control locally, but that part
cannot run inside Streamlit Cloud.
"""
)


def draw_landmarks(image_bgr: np.ndarray, results) -> np.ndarray:
    annotated = image_bgr.copy()
    if not results.multi_hand_landmarks:
        return annotated

    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
        )

    return annotated


with st.sidebar:
    st.header("Try it")
    source_mode = st.radio("Input source", ["Camera snapshot", "Upload image"], index=0)
    st.info("For the full mouse and typing experience, run the local desktop app.")


image_file = st.camera_input("Capture a frame") if source_mode == "Camera snapshot" else st.file_uploader("Upload a hand image", type=["png", "jpg", "jpeg"])

if image_file is None:
    st.stop()

file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    st.error("Could not read the image.")
    st.stop()

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6,
)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = hands.process(image_rgb)
annotated = draw_landmarks(image_bgr, result)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original")
    st.image(image_rgb, use_container_width=True)

with col2:
    st.subheader("Detected hands")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
st.success(f"Detected {hand_count} hand(s).")