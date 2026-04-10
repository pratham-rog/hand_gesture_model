import cv2
import mediapipe as mp
import math
import numpy as np
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController, Button as MouseButton

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow('Virtual Keyboard', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Keyboard', 1600, 900)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, max_num_hands=2) 
mpDraw = mp.solutions.drawing_utils

keyboard = Controller()
os_mouse = MouseController()

screen_w, screen_h = 1280, 832
frameR = 100
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 5
is_mouse_held = False

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

class Button():
    def __init__(self, pos, text, size=[50, 50]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

x_start = 640
y_start = 180

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        x_pos = 60 * j + x_start
        y_pos = 60 * i + y_start
        buttonList.append(Button([x_pos, y_pos], key, size=[50, 50]))

space_w = 60 * 6 - 10
space_x = x_start + 60 * 2
space_y = y_start + 60 * 3
buttonList.append(Button([space_x, space_y], "Space", size=[space_w, 50]))

kb_x = x_start - 20
kb_y = y_start - 20
kb_w = 60 * 10 - 10 + 40
kb_h = 60 * 4 - 10 + 40

def draw_keyboard_bg(img, x, y, w, h):
    if y+h > img.shape[0] or x+w > img.shape[1] or y < 0 or x < 0: return
    roi = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(roi, (45, 45), 0)
    color_rect = np.full(roi.shape, (255, 255, 255), dtype=np.uint8)
    alpha = 0.05
    glass = cv2.addWeighted(blur, 1 - alpha, color_rect, alpha, 0)
    img[y:y+h, x:x+w] = glass
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

clicked = False
delay_counter = 0

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hovered_button = None
    is_kb_pinching = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            h, w, c = img.shape
            thumb = handLms.landmark[4]
            index = handLms.landmark[8]
            
            x1, y1 = int(index.x * w), int(index.y * h)
            x2, y2 = int(thumb.x * w), int(thumb.y * h)
            distance = math.hypot(x2 - x1, y2 - y1)
            
            if label == "Right":
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

            elif label == "Left":
                screen_x = np.interp(x1, (frameR, w - frameR), (0, screen_w))
                screen_y = np.interp(y1, (frameR, h - frameR), (0, screen_h))
                
                clocX = plocX + (screen_x - plocX) / smoothening
                clocY = plocY + (screen_y - plocY) / smoothening
                
                os_mouse.position = (clocX, clocY)
                plocX, plocY = clocX, clocY
                
                mouse_color = (255, 200, 0)
                if distance < 40:
                    mouse_color = (0, 0, 255)
                    if not is_mouse_held:
                        os_mouse.press(MouseButton.left)
                        is_mouse_held = True
                else:
                    if is_mouse_held:
                        os_mouse.release(MouseButton.left)
                        is_mouse_held = False
                        
                cv2.circle(img, (x1, y1), 10, mouse_color, cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, mouse_color, cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), mouse_color, 4)

    draw_keyboard_bg(img, kb_x, kb_y, kb_w, kb_h)

    for button in buttonList:
        bx, by = button.pos
        bw, bh = button.size
        
        is_hovered = (button == hovered_button)
        btn_clicked = (is_hovered and is_kb_pinching)
        
        alpha = 0.0
        if btn_clicked:
            color = (0, 255, 0)
            alpha = 0.4
        elif is_hovered:
            color = (255, 255, 255)
            alpha = 0.2
            
        if alpha > 0 and (by+bh <= img.shape[0] and bx+bw <= img.shape[1]):
            overlay = img[by:by+bh, bx:bx+bw].copy()
            color_rect = np.full(overlay.shape, color, dtype=np.uint8)
            blend = cv2.addWeighted(overlay, 1 - alpha, color_rect, alpha, 0)
            img[by:by+bh, bx:bx+bw] = blend
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        
        if button.text == "Space":
            text_x = bx + int(bw / 2) - 40
            cv2.putText(img, button.text, (text_x + 2, by + 34), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 2)
            cv2.putText(img, button.text, (text_x, by + 32), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        else:
            cv2.putText(img, button.text, (bx + 14, by + 36), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 2)
            cv2.putText(img, button.text, (bx + 12, by + 34), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        
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
cv2.destroyAllWindows()