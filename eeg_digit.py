import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import signal
import librosa

class EEGCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(EEGCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self._to_linear = None
        self._initialize_shape(torch.zeros(1, *input_shape))

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _initialize_shape(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EEGDigitModel:
    def __init__(self, model_path=None, num_classes=10, device='cpu'):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.eeg_sensors = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
                            'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        self.model = EEGCNN(input_shape=(14, 129, 129), num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, x):
        df = pd.DataFrame(x, columns=self.eeg_sensors)
        filtered_data = self.apply_filters(df[self.eeg_sensors].values.T)
        spectrogram = self.create_spectrogram(filtered_data)
        return torch.FloatTensor(spectrogram).unsqueeze(0)

    def apply_filters(self, eeg_signals, fs=128):
        lowcut, highcut = 1, 50
        nyquist = 0.5 * fs
        b, a = signal.butter(N=4, Wn=[lowcut/nyquist, highcut/nyquist], btype='band')
        return signal.filtfilt(b, a, eeg_signals)

    def create_spectrogram(self, data, fs=128, nperseg=128, noverlap=64):
        spectrograms = []
        for channel_data in data:
            f, t, Sxx = signal.spectrogram(channel_data, fs, nperseg=nperseg, noverlap=noverlap)
            spectrograms.append(10 * np.log10(Sxx))
        return np.array(spectrograms)

    def __call__(self, x):
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model using load_model() method.")
        
        x_preprocessed = self.preprocess(x)
        x_preprocessed = x_preprocessed.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x_preprocessed)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.item()