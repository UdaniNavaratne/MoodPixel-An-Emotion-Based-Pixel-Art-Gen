# app.py
import streamlit as st
from textblob import TextBlob
from PIL import Image, ImageDraw
import random

# App title
st.set_page_config(page_title="🎮 MoodPixel", layout="centered")
st.markdown("<h1 style='text-align:center;'>🎮 MoodPixel</h1>", unsafe_allow_html=True)

# User input
user_input = st.text_input("How are you feeling today?")

if user_input:
    # Analyze sentiment
    sentiment = TextBlob(user_input).sentiment.polarity
    mood = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    st.write(f"Detected mood: **{mood}**")

    # Generate 10x10 pixel art
    img = Image.new("RGB", (100, 100))
    draw = ImageDraw.Draw(img)

    for i in range(10):
        for j in range(10):
            if mood == "Positive":
                color = (random.randint(150, 255), random.randint(150, 255), random.randint(0, 100))
            elif mood == "Negative":
                color = (random.randint(0, 100), random.randint(0, 100), random.randint(100, 200))
            else:
                color = (150, 150, 150)
            draw.rectangle([i*10, j*10, i*10+10, j*10+10], fill=color)

    # Show the image (scaled up)
    img = img.resize((300, 300), resample=Image.NEAREST)
    st.image(img, caption=f"{mood} Pixel Art",use_container_width=False)
