from flask import Flask, request, render_template
import numpy as np
import pickle, os, gzip, shutil
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)

# ---------- Load compressed model ----------
if not os.path.exists("image_caption_model.h5"):
    with gzip.open("image_caption_model.h5.gz", 'rb') as f_in:
        with open("image_caption_model.h5", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
model = load_model("image_caption_model.h5")

# ---------- Load compressed tokenizer ----------
with gzip.open("tokenizer.pkl.gz", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34
vocab_size = len(tokenizer.word_index) + 1

# ---------- Pretrained VGG16 for feature extraction ----------
vgg = VGG16()
vgg = tf.keras.Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

# ---------- Helper functions ----------
def index_to_word(index, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return None

def extract_features(image_path):
    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = vgg.predict(img, verbose=0)
    return feature

def predict_caption(model, image_feature, tokenizer, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([image_feature, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word(yhat, tokenizer)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join("static", file.filename)
        os.makedirs("static", exist_ok=True)
        file.save(filepath)

        feature = extract_features(filepath)
        caption = predict_caption(model, feature, tokenizer, max_length)
        return render_template("index.html", caption=caption, img_path=filepath)

    return render_template("index.html", caption=caption)

if __name__ == "__main__":
    app.run(debug=True)
