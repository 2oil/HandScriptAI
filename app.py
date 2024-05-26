from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
import base64

app = Flask(__name__)
model = load_model('NewJeans2.h5', compile=True)
ascii_map = pd.read_csv("mapping.csv")

@app.route('/')
def index():
    return render_template("main.html")

@app.route('/predict', methods=["POST"])
def get_image():
    canvasdata = request.form['canvasimg']
    encoded_data = canvasdata.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    gray_image = gray_image / 255.0
    
    gray_image = np.expand_dims(gray_image, axis=-1)
    # 역transpose 추가
    gray_image = np.transpose(gray_image, (1, 0, 2))
    
    img = np.expand_dims(gray_image, axis=0)

    print('Image received: {}'.format(img.shape))
    prediction = model.predict(img)
    cl = list(prediction[0])
    predicted_char = ascii_map["Character"][cl.index(max(cl))]

    return jsonify(value=predicted_char)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
