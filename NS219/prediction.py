#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 05 09:00:00 2024
@author: ike
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


"""
Helper functions to instantiate, train, and run inference with recurrent neural
networks. Stock price prediction is a common task for those interested in deep
learning -- many online tutorials walk the novice through the process. The code
described here is largely modified from exercises described from the following
sources:
    -   kaggle.com/code/taronzakaryan/predicting-stock-price-using-lstm-model
        -pytorch
    -   github.com/Daammon/Stock-Prediction-with-RNN-in-Pytorch/blob/master/
        Rnn0.ipynb
    -   medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632
    -   jovian.com/nagendhiran-r/predicting-stock-price-using-pytorch
"""


class StockDataset(Dataset):
    def __init__(self, dataframe, features, window):
        super(StockDataset, self).__init__()
        self.data = dataframe[features].values.astype(float)
        self.window = window

    def __len__(self):
        return self.data.shape[0] - self.window

    def __getitem__(self, idx):
        sample = self.data[idx:idx + self.window]
        sample = (torch.tensor(sample[:-1]), torch.tensor(sample[-1]))
        return sample


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden = hidden_dim
        self.layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.layers, x.size(0), self.hidden).detach().double()
        c0 = torch.zeros(
            self.layers, x.size(0), self.hidden).detach().double()
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


def training_loop(model, dataset, save, size, epochs, lr=0.01):
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=size, shuffle=True)
    epoch_loss = []
    model = model.double()
    model.train()
    for e in tqdm(range(epochs)):
        batch_loss = 0
        for batch in loader:
            prediction = model(batch[0].double())
            loss = criterion(prediction, batch[1].double())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            batch_loss += loss.item()

        epoch_loss += [batch_loss / len(loader)]

    torch.save(model.state_dict(), save)
    return epoch_loss


def test_loop(model, load, seed, window, iterations, p_idx=0):
    with torch.no_grad():
        model = model.double()
        model.load_state_dict(torch.load(load))
        model.eval()
        for i in range(iterations):
            output = model(torch.tensor(
                seed[i: window + i][None]).double()).detach().numpy()
            seed[window + i, p_idx] = output[-1, p_idx]

    return seed[-iterations:]


