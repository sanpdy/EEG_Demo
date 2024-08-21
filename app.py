from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import random
import io
import numpy as np
import threading
import re
from eeg_digit import EEGDigitModel
#from examples.read_and_export_mne import record

app = Flask(__name__)
record_thread = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    global record_thread
    if record_thread and record_thread.is_alive():
        return jsonify(success=False, message="Recording already in progress")
    
    record_thread = threading.Thread(target=start_record_thread)
    record_thread.start()
    return jsonify(success=True, message="Starting recording process")

def start_record_thread():
    try:
        pass
    except Exception as e:
        print(f"Error in recording: {str(e)}")

@app.route('/stop_recording')
def stop_recording():
    global record_thread
    if record_thread and record_thread.is_alive():

        record_thread.join()
        return predict()
    else:
        return jsonify(success=False, message="No active recording to stop")

@app.route('/recording_status')
def recording_status():
    if record_thread and record_thread.is_alive():
        pass
        return jsonify(status="Recording")
    else:
        pass
        return jsonify(status="Not Recording")


@app.route('/predict')
def predict():
    recording_path = r'c:/Sankalp/EmotivCortex/recordings/image_28994_EPOC_229198_2024.08.02T13.11.17.07.00.csv'
    ids = {0: [6094, 45769, 43960], 1: [43948, 64303, 25416], 2: [12461, 67001, 47375],
       3: [59366, 47356, 7745], 4: [2874, 30708, 12306], 5: [28994, 59956, 14670],
       6: [48033, 67512, 37990], 7: [45973, 16712, 69543], 8: [3493, 31182, 12928],
       9: [40341, 35732, 25371]}
    
    id_to_label = {id_num: label for label, id_list in ids.items() for id_num in id_list}

    try:
        print('making df')
        first_row = pd.read_csv(recording_path, nrows=1, header=None)
        id_info = first_row.iloc[0,0]

        match = re.search(r'image_(\d+)', id_info)
        id_number = int(match.group(1))
        label = id_to_label.get(id_number, -1)
        df = pd.read_csv(recording_path, skiprows=1, header=0)
        df['Label'] = label
        print('made df')
        print(df.columns)
        print('making model')
        model = EEGDigitModel()
        print('made model')

        model.preprocess(df)
        print(df.columns)
        print('loading model')
        model.load_model('eeg_model.pth')
        print('loaded model')

        print('predicting')
        predictions = model.predict(df)
        print("True Label: ", label)
        print("Prediction:", predictions[0])
        
        return jsonify(success=True, prediction=str(predictions[0]))

    except Exception as e:
        print("Failed to gather prediction.")
        return jsonify(success=False, error=str(e))


if __name__ == '__main__':
    app.run(debug=True)