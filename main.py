import webview
from threading import Thread
import subprocess

def start_flask():
    subprocess.Popen(["python", "app.py"])

flask_thread = Thread(target=start_flask)
flask_thread.start()

webview.create_window("EEG Measuring", "http://127.0.0.1:5000/")
webview.start()
