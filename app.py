# app.py
import streamlit as st
from PIL import Image
import numpy as np

# Mood to color mapping
mood_color_map = {
    "happy": (255, 223, 0),
    "sad": (70, 130, 180),
    "angry": (220, 20, 60),
    "neutral": (200, 200, 200),
    "anxious": (123, 104, 238),
    "excited": (255, 105, 180),
    "tired": (176, 196, 222)
}

# Function to create pixel art image
def generate_pixel_art(mood: str, size: int = 10):
    mood = mood.lower().strip()
    color = mood_color_map.get(mood, (128, 128, 128))  # Default gray
    pixels = np.full((size, size, 3), color, dtype=np.uint8)
    img = Image.fromarray(pixels, 'RGB')
    return img.resize((200, 200), resample=Image.Resampling.NEAREST)

# --- Streamlit UI ---
st.set_page_config(page_title="🌱 MoodPixel - Emotion-to-Pixel Art Generator", layout="centered")
st.title("🌱 MoodPixel")
st.subheader("Turn your mood into pixel art 🎨")

mood_input = st.text_input("Enter your mood (e.g., happy, sad, tired, excited):")

if mood_input:
    pixel_art = generate_pixel_art(mood_input)
    st.image(pixel_art, caption=f"Mood: {mood_input.capitalize()}")

    # Save option
    if st.button("Download Pixel Art"):
        pixel_art.save("mood_pixel_art.png")
        with open("mood_pixel_art.png", "rb") as file:
            st.download_button(label="Click to download", data=file, file_name="mood_pixel_art.png", mime="image/png")
