import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os
from groq import Groq
from gtts import gTTS
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

# CONFIG
IMG_SIZE = (224, 224)

client = Groq(api_key=GROK_API_KEY)  # api key

# Load class names from animals folder
DATASET_PATH = "animals"
CLASS_NAMES = sorted(os.listdir(DATASET_PATH))

# Load model
model = load_model("model/animal_model.h5")



def get_animal_info(animal_name):
    try:
        prompt = f"""
Write a short, friendly description of the animal: {animal_name}.

Structure:
1. Start with 2–3 simple sentences describing the animal
2. Then add "Key facts" section
3. Include emojis
4. Use bullet points for key facts
5. Keep language simple and easy to understand

Include:
- Scientific name
- Habitat
- Diet
- Lifespan
- Distinctive feature
- Social behavior (if applicable)
- Location (if known)

Rules:
- Do NOT include history
- Do NOT include long paragraphs
- Keep it concise
- Make it engaging
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a wildlife expert writing simple educational descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error fetching info: {str(e)}"

def text_to_speech(text):
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# UI
st.title("🐾 Animal Classification App")

uploaded_file = st.file_uploader(
    "Upload Animal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Animal: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Groq info
    with st.spinner("Fetching animal information..."):
        info = get_animal_info(predicted_class)

    st.subheader("📚 Animal Information")
    st.write(info)

    if st.checkbox("🔊 Listen to voice reply"):
        audio_file = text_to_speech(info)
        st.audio(audio_file)

