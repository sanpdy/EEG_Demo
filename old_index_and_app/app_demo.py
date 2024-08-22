from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import random
import io
import numpy as np
import threading
import re
from eeg_digit import EEGDigitModel
import os
from examples.read_and_export_mne import record
import mne
from model_inference import ModelInference, fif_to_dataframe

app = Flask(__name__)
record_thread = None

prediction_recording_path = ""


def fif_to_dataframe(fif_file):
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    data, times = raw[:]
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df["time"] = times
    return df


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
    output_dir = f"/home/xnguyen/projects/eeg_projects/eeg_demo/new_recordings/demo/"
    os.makedirs(output_dir, exist_ok=True)
    try:
        global prediction_recording_path
        prediction_recording_path = record(output_dir)
        stop_recording()
    except Exception as e:
        print(f"Error in recording: {str(e)}")


@app.route("/stop_recording")
def stop_recording():
    global record_thread
    if record_thread and record_thread.is_alive():

        record_thread.join()
        return predict()
    else:
        return jsonify(success=False, message="No active recording to stop")


@app.route("/recording_status")
def recording_status():
    if record_thread and record_thread.is_alive():
        pass
        return jsonify(status="Recording")
    else:
        pass
        return jsonify(status="Not Recording")


MODEL_CHECKPOINT = "model.pth"


@app.route("/predict")
def predict():
    recording_path = prediction_recording_path

    try:
        print("Making DataFrame")
        df = fif_to_dataframe(recording_path)
        print("DataFrame created")
        print("Initializing model")
        model_inference = ModelInference(checkpoint=MODEL_CHECKPOINT)
        print("Model initialized")

        # Prediction
        print("Preprocessing data")
        predictions = model_inference(
            df
        )  # Use the ModelInference instance for prediction
        print("Prediction:", predictions)

        return jsonify(success=True, prediction=str(predictions))

    except Exception as e:
        print("Failed to gather prediction.")
        return jsonify(success=False, error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
