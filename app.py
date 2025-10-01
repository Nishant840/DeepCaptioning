import os
import numpy as np
import pickle, gzip
from flask import Flask, request, render_template
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# -------- Load trained model --------
model_path = "image_caption_model_inference.h5"
model = load_model(model_path, compile=False)

# -------- Load tokenizer & max_length --------
with gzip.open("tokenizer.pkl.gz", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 35

# Ensure startseq/endseq exist in tokenizer
if "startseq" not in tokenizer.word_index:
    tokenizer.word_index["startseq"] = len(tokenizer.word_index) + 1
if "endseq" not in tokenizer.word_index:
    tokenizer.word_index["endseq"] = len(tokenizer.word_index) + 1

# -------- Load precomputed image features --------
with open("image_features.pkl", "rb") as f:
    image_features = pickle.load(f)  # dict: {image_id: feature_array of shape (1,4096)}

# -------- Helper functions --------
def index_to_word(index, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return None

def predict_caption(model, image_feature, tokenizer, max_length):
    in_text = "startseq"

    # feature already shape (1,4096)
    feature = np.array(image_feature, dtype='float32')

    for _ in range(max_length):
        # prepare sequence
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_length, padding='post')
        
        # predict next word
        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat, axis=-1)[0]  # get predicted index
        
        word = index_to_word(yhat, tokenizer)
        if not word or word == "endseq":
            break
        in_text += " " + word

    # remove startseq and endseq
    final = in_text.split()[1:]  # remove startseq
    if final[-1:] == ["endseq"]:
        final = final[:-1]
    return " ".join(final)

# -------- Flask routes --------
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    filepath = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join("static", file.filename)
        os.makedirs("static", exist_ok=True)
        file.save(filepath)

        # get feature from precomputed features
        image_id = os.path.splitext(file.filename)[0]
        if image_id not in image_features:
            return render_template("index.html", caption="❌ Feature not found for this image", img_path=filepath)

        feature = image_features[image_id]  # shape (1,4096)
        caption = predict_caption(model, feature, tokenizer, max_length)
        return render_template("index.html", caption=caption, img_path=filepath)

    return render_template("index.html", caption=caption, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
