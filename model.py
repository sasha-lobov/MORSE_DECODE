import librosa
import numpy as np
import torch
import torch.nn as nn
import json

class MorseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
        )
        # После двух пуллингов размер сигнала: 345 -> ~86 (точнее: floor(345 / 4) = 86)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(128 * 2, num_classes + 1)  # +1 для CTC blank

    def forward(self, x):
        x = self.cnn(x)              # (B, C, L)
        x = x.permute(0, 2, 1)       # (B, L, C)
        x, _ = self.gru(x)           # (B, L, 2*H)
        x = self.classifier(x)       # (B, L, C+1)
        return x.log_softmax(dim=-1) # (B, L, C+1)


def predict_single(signal_path, model, idx2char):
    model.eval()
    device = next(model.parameters()).device

    # Загрузка и подготовка сигнала
    signal = load_signal(signal_path)
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 345)
    signal = signal.to(device)

    with torch.no_grad():
        logits = model(signal)              # (1, T, C+1)
        pred = logits.argmax(dim=-1)        # (1, T)
        pred = pred.squeeze(0).cpu().numpy().tolist()

    # CTC-декодинг (удаление повторов и blank-индексов)
    blank_idx = len(idx2char)
    decoded = []
    prev = -1
    for p in pred:
        if p != prev and p != blank_idx:
            decoded.append(p)
        prev = p

    return ''.join([idx2char[i] for i in decoded])


def norm_signal(y, plot=False):

    min_val = np.min(y)
    max_val = np.max(y)
    
    if max_val == min_val:
        return np.zeros_like(y)  # если сигнал плоский — просто нули

    y_norm = 2 * (y - min_val) / (max_val - min_val) - 1
    x_norm = np.linspace(0, 8, num=len(y))
    
    if plot is True:
        plt.figure(figsize=(10, 4))
        plt.plot(x_norm, y_norm, color='red')
        plt.title("после нормализации")
        plt.show()
        
    return y_norm


def create_vocab(char_list):
    char2idx = {c: i for i, c in enumerate(char_list)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def load_signal(path):
    with open(path, "r") as f:
        y = json.load(f)
    return np.array(y, dtype=np.float32)

def save_signal(y, path):

    serializable = y.tolist()
    
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)