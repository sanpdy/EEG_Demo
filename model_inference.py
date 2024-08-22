from torch.utils.data import Dataset
import torch.nn as nn
import timm
import numpy as np
import cv2
import mne
import pandas as pd
from scipy.signal import butter, lfilter
import torch


def butter_bandpass_filter(
    data, high_freq=20, low_freq=0.5, sampling_rate=256, order=2
):
    nyquist = 0.5 * sampling_rate
    high_cutoff = high_freq / nyquist
    low_cutoff = low_freq / nyquist
    b, a = butter(order, [low_cutoff, high_cutoff], btype="band", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def fif_to_dataframe(fif_file):
    # Read the FIF file
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

    # Get the data and times
    data, times = raw[:, :]

    # Create a DataFrame
    df = pd.DataFrame(data.T, columns=raw.ch_names)

    # Add a time column
    df["time"] = times

    return df


class EEGDigitModel(nn.Module):
    def __init__(self, model_name, in_chans, num_classes):
        super(EEGDigitModel, self).__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x):
        return self.model(x)


class ModelInference:

    def __init__(self, checkpoint, device="cuda"):
        self.model = EEGDigitModel(model_name="resnet18", in_chans=14, num_classes=10)
        state_dict = torch.load(checkpoint)["model"]
        fixed_state_dict = {}
        for k, v in state_dict.items():
            kk = k.replace("module.", "")
            fixed_state_dict[kk] = v

        self.device = torch.device(device)
        self.model.load_state_dict(fixed_state_dict)
        self.model.eval()
        self.model.to(self.device)
        self.input_size = (224, 224)

    def preprocess(self, raw_data):
        from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

        spec_args = dict(
            sample_rate=256,
            n_fft=1024,
            n_mels=128,
            f_min=0.53,
            f_max=24,
            win_length=128,
            hop_length=39,
        )
        transform = nn.Sequential(MelSpectrogram(**spec_args), AmplitudeToDB())
        all_channels = raw_data.columns

        mfccs = []
        for channel in all_channels:
            if channel in ["TRIGGER", "time"]:
                continue

            channel_signal = raw_data[channel].values.reshape(
                -1,
            )
            channel_signal = channel_signal[: len(channel_signal) // 2]
            channel_signal = butter_bandpass_filter(channel_signal)
            channel_signal = torch.from_numpy(channel_signal).float().reshape(1, -1)
            mfcc = transform(channel_signal).cpu().numpy()[0]
            mfcc = cv2.resize(mfcc, self.input_size)
            mfccs.append(mfcc)

        mfccs = np.stack(mfccs, 0)  # Num_channels x H x W
        mfccs /= mfccs.max()
        mfccs = torch.from_numpy(mfccs).unsqueeze(0).to(self.device)
        return mfccs

    def __call__(self, raw_data):
        x = self.preprocess(raw_data)
        with torch.no_grad():
            output = self.model(x)

        _, prediction = output.max(1)
        return prediction.data[0]


if __name__ == "__main__":
    raw_data = fif_to_dataframe("new_recordings/image_2874/data_20240820_235934.fif")
    model_inference = ModelInference(checkpoint="logs/best.pth")
    output = model_inference(raw_data=raw_data)
    print(output)
