import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd


class EEGNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EEGNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EEGDigitModel:
    def __init__(self, input_size=14, num_classes=10, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = EEGNet(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

        self.eeg_sensors = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
                            'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

        self.cq_columns = ['EEG.RawCq']

        self.mean = None
        self.std = None

    def preprocess(self, df):
        # Calculate the percentage of good quality readings for each row
        df['CQ_percentage'] = df[self.cq_columns].apply(lambda x: (x >= 0.83).mean(), axis=1)

        high_quality_df = df[df['CQ_percentage'] >= 1]

        X = high_quality_df[self.eeg_sensors].values
        y = high_quality_df['Label'].values
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        if self.mean is None or self.std is None:
            self.mean = X.mean(dim=0)
            self.std = X.std(dim=0)

        X = (X - self.mean) / self.std

        return X, y

    def train(self, df, num_epochs=50, batch_size=64):
        X, y = self.preprocess(df)

        dataset = EEGDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        self.evaluate(test_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(batch_y.numpy())

        accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
        print(f"Accuracy: {accuracy:.4f}")

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame with the necessary EEG and CQ columns")

        X['CQ_percentage'] = X[self.cq_columns].apply(lambda x: (x >= 0.83).mean(), axis=1)
        high_quality_X = X[X['CQ_percentage'] >= 1]

        if high_quality_X.empty:
            raise ValueError("No high-quality data points found in the input")

        X_features = high_quality_X[self.eeg_sensors].values
        X_tensor = torch.FloatTensor(X_features)
        X_normalized = (X_tensor - self.mean) / self.std

        self.model.eval()
        with torch.no_grad():
            X_normalized = X_normalized.to(self.device)
            outputs = self.model(X_normalized)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.model.eval()
