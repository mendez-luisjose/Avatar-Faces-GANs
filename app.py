import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from utils import set_background

set_background("./imgs/background.png")

header = st.container()
body = st.container()

model=''

if model=='':
    model = tf.keras.models.load_model(
        ("./model/avatar_faces_model.h5"),
        custom_objects={'KerasLayer':hub.KerasLayer}
    )


def generate_avatar_images(model) :
    test_input = tf.random.normal([16, 128])
    n = 16
    n = int(np.sqrt(n))

    predictions = model.predict(test_input)

    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(6, 6))

    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(predictions[i], cmap="viridis")

    return fig
    

with header :
    _, col0, _ = st.columns([0.25,1,0.1])
    col0.title("💥 GANs Avatar Face Generator 🤖")

    _, col1, _ = st.columns([0.3,1,0.2])
    col1.image("./imgs/avatars_gan.gif", width=400)

    _, col2, _ = st.columns([0.3,1,0.2])
    col2.subheader("Avatar Face Model Generator with TensorFlow 🧪")

    _, col3, _ = st.columns([0.3,1,0.1])
    col3.image("./imgs/avatars_preview.png", width=370)

    st.write("This GANs Model was trained with over 20.000 Images, using TensorFlow and the Google Colab GPU.")

with body :
    _, col4, _ = st.columns([0.4,1,0.3])
    col4.subheader("Check It-out the GANs Generator Model 🔎!")

    _, col5, _ = st.columns([0.65,1,0.2])

    if col5.button("Generate Avatar Images"):
        avatars = generate_avatar_images(model)
        
        _, col6, _ = st.columns([0.5,1,0.2])
        col6.header("Avatar Results ✅:")

        _, col7, _ = st.columns([0.1,1,0.1])
        col7.pyplot(avatars)
