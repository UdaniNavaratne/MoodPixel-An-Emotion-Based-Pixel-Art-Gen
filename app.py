import streamlit as st
from PIL import Image
import numpy as np
import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
# === Mood-to-color map ===
mood_color_map = {
    "happy": (255, 223, 0),
    "sad": (70, 130, 180),
    "angry": (220, 20, 60),
    "neutral": (200, 200, 200),
    "anxious": (123, 104, 238),
    "excited": (255, 105, 180),
    "tired": (176, 196, 222)
}

emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐",
    "anxious": "🫨", "excited": "🤩", "tired": "😴"
}

# === Pixel Art Generator ===
def generate_pixel_art(mood: str, size: int = 10):
    mood = mood.lower().strip()
    color = mood_color_map.get(mood, (128, 128, 128))
    pixels = np.full((size, size, 3), color, dtype=np.uint8)
    img = Image.fromarray(pixels, 'RGB')
    return img.resize((200, 200), resample=Image.Resampling.NEAREST)
def gpt_classify_mood(user_text):
    prompt = (
        "Classify the following text into a single mood for visual pixel art. "
        "Possible moods: happy, sad, angry, tired, excited, anxious, or neutral.\n"
        f"Text: \"{user_text}\"\nMood:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a helpful assistant that outputs only a one-word mood."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        mood = response.choices[0].message.content.strip().lower()
        return mood if mood in mood_color_map else "neutral"
    except Exception as e:
        st.error(f"Error classifying mood: {e}")
        return "neutral"

# === Streamlit App ===
st.set_page_config(page_title="🧠 MoodPixel + GPT", layout="centered")
st.title("🌱 MoodPixel + GPT")
st.subheader("Type your feelings — GPT decides the mood 🌈")

user_input = st.text_input("How are you feeling right now?")

if user_input:
    mood = gpt_classify_mood(user_input)
    pixel_img = generate_pixel_art(mood)
    emoji = emoji_map.get(mood, "")

    st.markdown(f"**GPT Mood:** `{mood}` {emoji}")
    st.image(pixel_img, caption=f"Pixel art for: {mood}")

    if st.button("Download this mood"):
        pixel_img.save("moodpixel_gpt.png")
        with open("moodpixel_gpt.png", "rb") as file:
            st.download_button("Download Image", file, file_name="moodpixel_gpt.png", mime="image/png")
