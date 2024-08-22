from datetime import datetime

import numpy as np
from mne import Info, create_info
from mne.io.array import RawArray
from pylsl import StreamInlet, resolve_stream
import os
from config import SRATE


def get_info() -> Info:
    ch_names = [
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
    ]

    info = create_info(sfreq=SRATE, ch_names=ch_names, ch_types=["eeg"] * len(ch_names))

    return info


def record(dir: str):
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream("type", "EEG")
    print("Found EEG stream.")
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    buffer = []
    while True:
        if len(buffer) == 128 * 2:  # wait 10 seconds
            break

        sample, _ = inlet.pull_sample()
        sample = [el / 1000000 for el in sample]  # convert to microvolts

        buffer.append(sample)

    info = get_info()
    raw = RawArray(np.array(buffer).T, info)

    # Generate a filename based on the current timestamp
    filename = "data_{}.fif".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(dir, filename)

    # Save the raw data to the specified path
    raw.save(path)

    print(f"Data saved to {path}")
    return path
