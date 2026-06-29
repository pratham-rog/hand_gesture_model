import os
import sys

# ── Suppress MediaPipe / TensorFlow log noise BEFORE importing them ────────────
os.environ["GLOG_minloglevel"] = "3"          # Suppress INFO/WARNING/ERROR logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Suppress TF C++ logs
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"     # Avoid GL/GPU init noise
os.environ["GRPC_VERBOSITY"] = "ERROR"
# Redirect stderr to suppress any remaining C++ noise during import
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarkerResult
import math
# pyrefly: ignore [missing-import]
import numpy as np
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController, Button as MouseButton
import urllib.request
import threading

# Restore stderr after noisy imports are done
sys.stderr = _stderr

# ── Download model if not present ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model (~10 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow('Virtual Keyboard', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Keyboard', 1600, 900)

# ── Controllers ────────────────────────────────────────────────────────────────
keyboard = Controller()
os_mouse = MouseController()

screen_w, screen_h = 1280, 832
frameR = 100
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothing = 5

PINCH_CONFIRM_FRAMES = 5
CLICK_COOLDOWN_FRAMES = 20

pinch_frame_count = 0
mouse_click_cooldown = 0
mouse_was_pinching = False

# ── Keyboard layout ────────────────────────────────────────────────────────────
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

class Button:
    def __init__(self, pos, text, size=[50, 50]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
x_start, y_start = 640, 180

for i, row in enumerate(keys):
    for j, key in enumerate(row):
        buttonList.append(Button([60 * j + x_start, 60 * i + y_start], key))

space_w = 60 * 6 - 10
buttonList.append(Button([x_start + 60 * 2, y_start + 60 * 3], "Space", size=[space_w, 50]))

kb_x, kb_y = x_start - 20, y_start - 20
kb_w = 60 * 10 - 10 + 40
kb_h = 60 * 4 - 10 + 40

def draw_keyboard_bg(img, x, y, w, h):
    if y+h > img.shape[0] or x+w > img.shape[1] or y < 0 or x < 0:
        return
    roi = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(roi, (45, 45), 0)
    color_rect = np.full(roi.shape, (255, 255, 255), dtype=np.uint8)
    glass = cv2.addWeighted(blur, 0.95, color_rect, 0.05, 0)
    img[y:y+h, x:x+w] = glass
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

# ── MediaPipe Tasks setup ──────────────────────────────────────────────────────
latest_result: HandLandmarkerResult = None
result_lock = threading.Lock()

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8,
    result_callback=result_callback,
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

# ── Drawing connections ────────────────────────────────────────────────────────
HAND_CONNECTIONS = mp_vision.HandLandmarker.HAND_CONNECTIONS if hasattr(mp_vision.HandLandmarker, 'HAND_CONNECTIONS') else None

def draw_hand(img, landmarks_norm, h, w):
    """Draw hand skeleton on the image."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),         # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (0,9),(9,10),(10,11),(11,12),     # middle
        (0,13),(13,14),(14,15),(15,16),   # ring
        (0,17),(17,18),(18,19),(19,20),   # pinky
        (5,9),(9,13),(13,17),             # palm
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm]
    for a, b in connections:
        cv2.line(img, pts[a], pts[b], (200, 200, 200), 1)
    for pt in pts:
        cv2.circle(img, pt, 4, (255, 255, 255), cv2.FILLED)

# ── Main loop ──────────────────────────────────────────────────────────────────
clicked = False
delay_counter = 0
frame_ts = 0

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # Send frame to MediaPipe (async)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 1
    landmarker.detect_async(mp_img, frame_ts)

    # Read latest result
    with result_lock:
        result = latest_result

    hovered_button = None
    is_kb_pinching = False

    if result and result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # Handedness: "Left"/"Right" (already mirrored-aware in Tasks API)
            label = result.handedness[i][0].category_name if result.handedness else "Right"

            draw_hand(img, hand_landmarks, h, w)

            thumb  = hand_landmarks[4]
            index  = hand_landmarks[8]

            x1, y1 = int(index.x * w), int(index.y * h)
            x2, y2 = int(thumb.x * w), int(thumb.y * h)
            distance = math.hypot(x2 - x1, y2 - y1)

            if label == "Right":
                # Keyboard control hand
                for button in buttonList:
                    bx, by = button.pos
                    bw, bh = button.size
                    if bx < x1 < bx + bw and by < y1 < by + bh:
                        hovered_button = button
                        break

                pointer_color = (255, 255, 255)
                if distance < 40:
                    is_kb_pinching = True
                    pointer_color = (0, 255, 0)

                cv2.circle(img, (x1, y1), 8, pointer_color, cv2.FILLED)
                cv2.circle(img, (x2, y2), 8, pointer_color, cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), pointer_color, 3)

            else:
                # Mouse control hand
                screen_x = np.interp(x1, (frameR, w - frameR), (0, screen_w))
                screen_y = np.interp(y1, (frameR, h - frameR), (0, screen_h))
                clocX = plocX + (screen_x - plocX) / smoothing
                clocY = plocY + (screen_y - plocY) / smoothing
                os_mouse.position = (clocX, clocY)
                plocX, plocY = clocX, clocY

                mouse_color = (255, 200, 0)
                is_pinching_now = distance < 40

                if mouse_click_cooldown > 0:
                    mouse_click_cooldown -= 1

                if is_pinching_now:
                    pinch_frame_count += 1
                    mouse_color = (0, 100, 255)
                    if pinch_frame_count == PINCH_CONFIRM_FRAMES and mouse_click_cooldown == 0:
                        mouse_color = (0, 0, 255)
                        os_mouse.click(MouseButton.left, 1)
                        mouse_click_cooldown = CLICK_COOLDOWN_FRAMES
                    elif pinch_frame_count >= PINCH_CONFIRM_FRAMES:
                        mouse_color = (0, 0, 200)
                else:
                    pinch_frame_count = 0

                mouse_was_pinching = is_pinching_now
                cv2.circle(img, (x1, y1), 10, mouse_color, cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, mouse_color, cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), mouse_color, 4)

    # Draw keyboard
    draw_keyboard_bg(img, kb_x, kb_y, kb_w, kb_h)

    for button in buttonList:
        bx, by = button.pos
        bw, bh = button.size
        is_hovered = (button == hovered_button)
        btn_clicked = (is_hovered and is_kb_pinching)

        alpha = 0.0
        color = (255, 255, 255)
        if btn_clicked:
            color = (0, 255, 0)
            alpha = 0.4
        elif is_hovered:
            alpha = 0.2

        if alpha > 0 and (by+bh <= img.shape[0] and bx+bw <= img.shape[1]):
            overlay = img[by:by+bh, bx:bx+bw].copy()
            color_rect = np.full(overlay.shape, color, dtype=np.uint8)
            blend = cv2.addWeighted(overlay, 1 - alpha, color_rect, alpha, 0)
            img[by:by+bh, bx:bx+bw] = blend
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255, 255, 255), 1)

        if button.text == "Space":
            text_x = bx + int(bw / 2) - 40
            cv2.putText(img, button.text, (text_x+2, by+34), cv2.FONT_HERSHEY_PLAIN, 2, (100,100,100), 2)
            cv2.putText(img, button.text, (text_x,   by+32), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        else:
            cv2.putText(img, button.text, (bx+14, by+36), cv2.FONT_HERSHEY_PLAIN, 2, (100,100,100), 2)
            cv2.putText(img, button.text, (bx+12, by+34), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

        if btn_clicked and not clicked:
            if button.text == "Space":
                keyboard.press(Key.space)
                keyboard.release(Key.space)
            else:
                keyboard.press(button.text)
                keyboard.release(button.text)
            clicked = True
            delay_counter = 0

    if clicked:
        delay_counter += 1
        if delay_counter > 15:
            clicked = False

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()