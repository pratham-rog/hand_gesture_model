# Virtual Keyboard & Air Mouse System

A futuristic, completely touchless computer interaction system built with Python, OpenCV, and MediaPipe. 

This application uses your webcam to track both of your hands in real-time, assigning independent OS-level commands to each hand. You can navigate your computer, click elements, and type onto a virtual augmented keyboard entirely in thin air!

## Features 🚀

- **Dual Hand Multi-Modal Tracking:** Uses machine learning to detect and track your left and right hands individually. 
- **Physical Left Hand -> OS Mouse Controller:** Move your left hand in the air to effortlessly move your system mouse pointer. Features a coordinate smoothening layer for buttery stable, jitter-free precision. 
- **Physical Right Hand -> Augmented Keyboard:** A beautifully designed Virtual Keyboard rendered natively on your screen utilizing proper translucent Glassmorphism techniques.
- **Pinch-to-Click Dynamics:** Instead of hovering or awkward finger overlaps, this application utilizes the industry-trusted **Thumb to Index Finger PINCH** structure for executing Left Clicks and keyboard inputs.
- **Hold-to-Drag Capability:** Hold the pinch with your left hand to simulate dragging windows or highlighting text globally across your Mac.
- **System-Wide Typing (`pynput`):** The application types at the OS level! Leave it running in the background while you focus on Google Chrome or any text editor to type physically without touching a keyboard.

## Installation 🛠️

1. Activate your virtual environment: 
```bash
source venv/bin/activate
```
2. Install the necessary machine learning and system navigation dependencies:
```bash
pip install -r requirements.txt
```

## How to Run & Use 💻

1. Launch the application:
```bash
venv/bin/python key.py
```
2. Your camera will activate, presenting a large 1600x900 window containing the Virtual Keyboard on the right.
3. Bring your **left hand** in frame to control the mouse pointer. Pinch your thumb and index finger together to Left Click.
4. Bring your **right hand** in frame and hover your index finger over the augment keyboard keys. Pinch your thumb and index together to type the letter!
