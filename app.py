import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.title("Fruit & Vegetable Classifier")

st.write(
    "Upload an image of a fruit or vegetable and the model will predict its class."
)
@st.cache_resource
def load_my_model():
    return load_model('Image_classify.keras')

model = load_my_model()
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180
uploaded_file = st.file_uploader(
    "Upload Fruit/Vegetable Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_load = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(img_height, img_width)
    )

    img_arr = tf.keras.utils.img_to_array(image_load)

    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat, verbose=0)

    score = tf.nn.softmax(predict)

    st.image(uploaded_file, width=300)

    st.success(
        f"Prediction: {data_cat[np.argmax(score)]}"
    )

    confidence = float(np.max(score))

    st.write(f"Confidence: {confidence*100:.2f}%")
    st.progress(confidence)