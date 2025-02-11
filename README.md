# Install required libraries
!pip install gradio mediapipe opencv-python gtts numpy

# Import libraries
import cv2
import mediapipe as mp
import gradio as gr
import numpy as np
from gtts import gTTS

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Increased for better detection
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Define gestures (A to B, 1 to 9 included)
GESTURES = {
    "Thumbs Up": "Hello",
    "Open Palm": "Thank You",
    "Fist": "Yes",
    "Pointing Up": "No",
    "Victory": "Peace",
    "One": "1",
    "Two": "2",
    "Three": "3",
    "Four": "4",
    "Five": "5",
    "A": "A",
    "B": "B"
}

# Multi-language support
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr"
}

# Function to detect gestures
def detect_gesture(landmarks):
    """Detects gestures based on hand landmark positions."""
    def is_finger_extended(tip, pip, dip):
        return tip.y < pip.y and tip.y < dip.y  # Checks if a finger is extended

    # Assign landmark positions
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]

    # Gesture rules
    if is_finger_extended(thumb_tip, thumb_ip, thumb_ip) and not any(
        is_finger_extended(finger_tip, finger_pip, finger_dip)
        for finger_tip, finger_pip, finger_dip in [(index_tip, index_pip, index_dip), 
                                                   (middle_tip, middle_pip, middle_dip), 
                                                   (ring_tip, ring_pip, ring_dip), 
                                                   (pinky_tip, pinky_pip, pinky_dip)]
    ):
        return "Thumbs Up"

    if all(
        is_finger_extended(finger_tip, finger_pip, finger_dip)
        for finger_tip, finger_pip, finger_dip in [(index_tip, index_pip, index_dip), 
                                                   (middle_tip, middle_pip, middle_dip), 
                                                   (ring_tip, ring_pip, ring_dip), 
                                                   (pinky_tip, pinky_pip, pinky_dip)]
    ):
        return "Open Palm"

    if not any(
        is_finger_extended(finger_tip, finger_pip, finger_dip)
        for finger_tip, finger_pip, finger_dip in [(index_tip, index_pip, index_dip), 
                                                   (middle_tip, middle_pip, middle_dip), 
                                                   (ring_tip, ring_pip, ring_dip), 
                                                   (pinky_tip, pinky_pip, pinky_dip)]
    ):
        return "Fist"

    if is_finger_extended(index_tip, index_pip, index_dip) and not any(
        is_finger_extended(finger_tip, finger_pip, finger_dip)
        for finger_tip, finger_pip, finger_dip in [(middle_tip, middle_pip, middle_dip), 
                                                   (ring_tip, ring_pip, ring_dip), 
                                                   (pinky_tip, pinky_pip, pinky_dip)]
    ):
        return "Pointing Up"

    if is_finger_extended(index_tip, index_pip, index_dip) and is_finger_extended(middle_tip, middle_pip, middle_dip):
        return "Victory"

    extended_fingers = sum(
        is_finger_extended(finger_tip, finger_pip, finger_dip)
        for finger_tip, finger_pip, finger_dip in [(index_tip, index_pip, index_dip), 
                                                   (middle_tip, middle_pip, middle_dip), 
                                                   (ring_tip, ring_pip, ring_dip), 
                                                   (pinky_tip, pinky_pip, pinky_dip)]
    )

    return GESTURES.get(str(extended_fingers), "Unknown")

# Function to process webcam images
def process_image(image, language):
    """Processes image to detect gestures and return translation & speech output."""
    if image is None:
        return "No image detected", "", None

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert image format
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Detect hand landmarks

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw detected hand
            gesture = detect_gesture(hand_landmarks.landmark)
            translation = GESTURES.get(gesture, "Gesture Unknown")

            # Generate text-to-speech output
            tts = gTTS(text=translation, lang=LANGUAGES[language])
            tts.save("output.mp3")

            return gesture, translation, "output.mp3"

    return "No Hand Detected", "", None

# Gradio UI
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(sources="webcam"),  
        gr.Radio(list(LANGUAGES.keys()), label="Select Language")
    ],
    outputs=[
        gr.Textbox(label="Detected Gesture"),
        gr.Textbox(label="Translated Text"),
        gr.Audio(label="Audio Output")
    ],
    live=True,
    title="🖐️ AI Sign Language Translator – Real-Time Gesture to Speech & Text",
    description="An advanced AI-powered sign language translator that detects hand gestures in real-time and converts them into text and speech. Simply show a hand gesture, and the model will instantly recognize it, translate it into your selected language, and generate audio output. Perfect for seamless communication! 🌍🔊"
)

# Run the app
demo.launch()
