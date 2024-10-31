import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class FastTextDataset(Dataset):
    def __init__(self, X: list, y: np.ndarray, max_seq_length: float = 400):
        self.X = X
        self.y = y
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = torch.tensor(self.X[index]).float()
        y = torch.tensor(self.y[index]).float()
        seq_length = torch.tensor(self.X[index].shape[0])

        X = nn.functional.pad(X, (0, 0, 0, self.max_seq_length - X.shape[0]))
        return X, y, seq_length


class LSTMNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
            dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedded, text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_outputs = self.fc(hidden)
        return self.sigmoid(dense_outputs)
