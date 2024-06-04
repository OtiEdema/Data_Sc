import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

# Function to add a background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://www.transparenttextures.com/patterns/cubes.png");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Function to add header image
def add_header_image(image_path):
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_str}" style="width: 100%; height: auto; max-width: 600px;">
            </div>
            """,
            unsafe_allow_html=True
        )

# Load the trained model
model = load_model('brain_tumor_detector.h5')

# Function to make predictions
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0][0]

# Streamlit app
st.title("Brain Tumor Detection from MRI Images")

# Add background and header image
add_bg_from_url()
add_header_image("header_image.png")  # Replace with your header image path

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #00FF00;
        font-family: "Courier New", Courier, monospace;
    }
    .stButton>button {
        background-color: #00FF00;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #45a049;
    color:#ffffff;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img_path = "uploaded_image.png"
    img.save(img_path)
    prediction = model_predict(img_path, model)
    if prediction > 0.5:
        st.write("The image is classified as a Brain Tumor with {:.2f}% confidence.".format(prediction * 100))
    else:
        st.write("The image is classified as No Brain Tumor with {:.2f}% confidence.".format((1 - prediction) * 100))
