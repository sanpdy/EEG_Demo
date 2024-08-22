from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import random
import io
import numpy as np
import threading
import re
from eeg_digit import EEGDigitModel
from examples.read_and_export_mne import record
from PIL import Image
import os

app = Flask(__name__)

# Load the dataset
train_df = pd.read_csv("samples/mnist_train.csv")
test_df = pd.read_csv("samples/mnist_test.csv")
data = pd.concat([train_df, test_df], ignore_index=True)

# Initialize state
random_indices = []
for label in range(10):
    indices = data[data.iloc[:, 0] == label].sample(1, random_state=1).index.tolist()
    random_indices.extend(indices)

current_index = 0
current_image_name = ""
record_thread = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_recording")
def start_recording():
    global record_thread
    if record_thread and record_thread.is_alive():
        return jsonify(success=False, message="Recording already in progress")

    record_thread = threading.Thread(target=start_record_thread)
    record_thread.start()
    return jsonify(success=True, message="Starting recording process")


def start_record_thread():
    global current_image_name

    output_dir = f"/home/xnguyen/projects/eeg_projects/eeg_demo/2_second_recordings/{current_image_name}/"
    os.makedirs(output_dir, exist_ok=True)
    try:
        record(output_dir)
    except Exception as e:
        print(f"Error in recording: {str(e)}")


@app.route("/stop_recording")
def stop_recording():
    global record_thread
    if record_thread and record_thread.is_alive():
        record.stop_record()
        record_thread.join()
        return jsonify(success=True, message="Recording stopped")
    else:
        return jsonify(success=False, message="No active recording to stop")


@app.route("/recording_status")
def recording_status():
    if record_thread and record_thread.is_alive():
        return jsonify(status="Recording", headset_found=record.headset_found)
    else:
        return jsonify(status="Not Recording", headset_found=record.headset_found)


@app.route("/previous_image")
def previous_image():
    global current_index, current_image_name
    current_index = (current_index - 1) % len(random_indices)
    current_image_name = f"image_{random_indices[current_index]}"
    return jsonify(success=True)


@app.route("/next_image")
def next_image():
    global current_index, current_image_name
    current_index = (current_index + 1) % len(random_indices)
    current_image_name = f"image_{random_indices[current_index]}"
    return jsonify(success=True)


@app.route("/current_image")
def current_image():
    global current_index, current_image_name
    row = data.iloc[random_indices[current_index]]
    pixels = row[1:].values.reshape(28, 28).astype(np.uint8)
    image = Image.fromarray(pixels, "L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    current_image_name = f"image_{random_indices[current_index]}"
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
