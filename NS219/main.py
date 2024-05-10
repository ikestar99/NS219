#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 05 09:00:00 2024
@author: ike
"""


import numpy as np
import pandas as pd
import os.path as op
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import NS219.graphs as ns219g
import NS219.prediction as ns219p


"""
Input parameters for analysis pipeline.
Full dataset is available on Kaggle at:
https://www.kaggle.com/datasets/parsabg/stocknewseventssentiment-snes-10/data

WARNING: You may need to modify file path browsing if working on a PC.
"""

"""
File names of raw data and template for saving figures.
"""
ROOT_FOLDER = "/Users/ike/Desktop/NS219 Project Extended"
DATA_PATH = "/".join((ROOT_FOLDER, "kaggle_stock_dataset.csv"))
FIGURE_PATH = "/".join((ROOT_FOLDER, "{}.png"))
MODEL_PATH = "/".join((ROOT_FOLDER, "{} LSTM Model.pt"))
PREDICTION_PATH = "/".join((ROOT_FOLDER, "Future Prediction Dataset.csv"))

"""
Relevant column headers in raw data.csv file.
"""
C_DATE = "Date"
C_OPEN = "Open"  # not used in this analysis
C_CLOSE = "Close"
C_VOLUME = "Volume"
C_SYMBOL = "Symbol"
C_NEWS = "News - Volume"
C_POSITIVE = "News - Positive Sentiment"
C_NEGATIVE = "News - Negative Sentiment"

"""
Columns that will be added to data.csv dataframe during analysis.
"""
C_CLASS = "Class"
C_CHANGE = "Δ Close Price, %"
C_SOURCE = "Source"
C_PREDICTION = "Predicted, %"

"""
Stock symbols to analyze from dataset.
"""
SUBSET = ["AAPL", "AMZN", "NFLX", "NVDA"]

"""
Stock % change >= THRESHOLD defines stocks that are good investments.
"""
THRESHOLD = 0
CLASS_LABELS = ["Depreciate", "Appreciate"]

"""
Degrees of freedom for z-score normalization
"""
DDOF = len(SUBSET) - 1

"""
Subset of data columns to use for different analyses.
"""
COLUMNS = [  # columns to extract from raw data.csv file
    C_DATE, C_CLOSE, C_VOLUME, C_SYMBOL, C_NEWS, C_POSITIVE, C_NEGATIVE]
LABELS = [  # columns by which to group datapoints
    C_SYMBOL, C_CLASS]
R_FEATURES = [  # variable columns to plot from raw data
    C_CLOSE, C_VOLUME, C_NEWS, C_POSITIVE, C_NEGATIVE]
N_FEATURES = [  # variable columns to plot from normalized data
    C_CHANGE] + R_FEATURES[1:]
P_FEATURES = R_FEATURES[1:]  # variable columns to cluster as predictive
LOG_NORM = [  # columns to normalize by taking the logarithm
    C_VOLUME, C_NEWS, C_POSITIVE, C_NEGATIVE]

"""
Parameters for LSTM network training and evaluation.
"""
WINDOW = 50  # WINDOW - 1 samples used to predict idx = WINDOW sample
N_BATCH = 100  # number of samples to predict before updating model
SPLIT = 0.7  # percentage of input data to use for training
N_INPUT = len(N_FEATURES)  # number of input features for LSTM network
N_HIDDEN = 32  # number of hidden features
N_LAYERS = 3  # number of recurrent layers
N_OUTPUT = N_INPUT  # number of output features for LSTM network
EPOCHS = 100  # number of epochs over which to train the network

"""
Parameters for dimensionality reduction techniques.
"""
N_COMPONENTS = 3
REDUCE = ["Dim {}".format(x) for x in range(N_COMPONENTS)]

"""
Style of seaborn backend for graphs.
"""
sns.set_theme(style="ticks")


def __main__():
    """
    Data preprocessing (normalization) and visualization.
    """
    # Load relevant features from data.csv file
    stock_df = pd.read_csv(DATA_PATH, usecols=COLUMNS).dropna(axis=0)
    stock_df = stock_df.loc[stock_df[C_SYMBOL].isin(SUBSET)].sort_values(
        by=[C_SYMBOL, C_DATE]).reset_index()
    stock_df[C_DATE] = pd.to_datetime(stock_df[C_DATE])

    # generate and save temporal line plots to visualize raw data
    save = FIGURE_PATH.format("Raw Data")
    ns219g.generate_pair_line(stock_df, save, [C_DATE], R_FEATURES, C_SYMBOL)

    # create dataframe of initial price value to be used later for prediction
    start_df = stock_df[[C_SYMBOL, C_DATE, C_CLOSE]].copy().groupby(
        [C_SYMBOL]).first().reset_index().rename(columns={C_CLOSE: C_CHANGE})

    # add relative price change column and drop closing price column
    stock_df = stock_df.assign(
        **{C_CHANGE: stock_df.groupby([C_SYMBOL])[C_CLOSE].pct_change()})
    stock_df = stock_df.dropna(axis=0).drop(columns=C_CLOSE).reset_index()

    # log normalize designated columns
    for c in LOG_NORM:
        stock_df[c] = np.log1p(stock_df[c])

    # z-score within-stock data for all features
    undo_z = (
        np.mean(stock_df[C_CHANGE]), np.std(stock_df[C_CHANGE], ddof=DDOF))
    for c in N_FEATURES:
        stock_df[c] = ss.zscore(stock_df[c], ddof=DDOF)
        # # hidden code to z-score within stock symbol
        # stock_df[c] = stock_df.groupby(
        #     [C_SYMBOL])[c].transform(ss.zscore, ddof=1)

    # generate and save temporal line plots to visualize normalized data
    save = FIGURE_PATH.format("Normalized Data")
    ns219g.generate_pair_line(stock_df, save, [C_DATE], N_FEATURES, C_SYMBOL)

    # label stocks as appreciating, depreciating, or no change by threshold
    conditions = [
        stock_df[C_CHANGE] < THRESHOLD, stock_df[C_CHANGE] >= -THRESHOLD]
    stock_df = stock_df.assign(
        **{C_CLASS: np.select(conditions, CLASS_LABELS)})

    # generate and save pairwise scatter plots to visualize labeled data
    save = FIGURE_PATH.format("Normalized Scatter Plots")
    ns219g.generate_pair_scatter(stock_df, save, P_FEATURES, C_SYMBOL, C_CLASS)

    """
    PCA: first pass dimensionality reduction method.
    """
    # perform PCA
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(stock_df[P_FEATURES])
    embed = pca.transform(stock_df[P_FEATURES])
    variance = pca.explained_variance_ratio_.reshape(1, -1)
    embed_df = pd.DataFrame(data=embed, columns=REDUCE).assign(
        **{c: stock_df[c] for c in LABELS})
    variance_df = pd.DataFrame(data=variance, columns=REDUCE, index=[0])

    # generate and save pairwise scatter plots to visualize transformed data
    save = FIGURE_PATH.format("PCA Scatter Plots")
    ns219g.generate_pair_scatter(embed_df, save, REDUCE, C_SYMBOL, C_CLASS)

    # generate fractional variance bar plot
    save = FIGURE_PATH.format("PCA Fractional Variance Bar Plot")
    if not op.isfile(save):
        ax = sns.barplot(variance_df)
        ax.set_ylabel("Fraction of Variance")
        sns.despine(ax=ax)
        ax.get_figure().savefig(save)
        plt.clf()

    # generate 3D plot to visualize first 3 components of PCA output
    for c in LABELS:
        save = FIGURE_PATH.format("3D PCA {} Scatter Plots".format(c))
        ns219g.generate_3d_plot(embed_df, save, REDUCE, c)

    """
    UMAP: second pass dimensionality reduction method.
    """
    # perform UMAP
    reducer = UMAP(n_components=N_COMPONENTS)
    embed = reducer.fit_transform(stock_df[P_FEATURES].values)
    embed_df = pd.DataFrame(data=embed, columns=REDUCE).assign(
        **{c: stock_df[c] for c in LABELS})

    # generate and save pairwise scatter plots to visualize transformed data
    save = FIGURE_PATH.format("UMAP Scatter Plots")
    ns219g.generate_pair_scatter(embed_df, save, REDUCE, C_SYMBOL, C_CLASS)

    # generate 3D plot to visualize first 3 components of PCA output
    for c in LABELS:
        save = FIGURE_PATH.format("3D UMAP {} Scatter Plots".format(c))
        ns219g.generate_3d_plot(embed_df, save, REDUCE, c)

    """
    Recurrent Neural Networks: third pass future prediction method.
    """
    # train the long short term memory recurrent neural networks
    model = ns219p.LSTM(N_INPUT, N_HIDDEN, N_LAYERS, N_OUTPUT)
    loss_df = pd.DataFrame()
    for s in SUBSET:
        save = MODEL_PATH.format(s)
        if op.isfile(save):
            continue

        subset_df = stock_df.loc[stock_df[C_SYMBOL] == s].copy()
        train_split = int(subset_df.shape[0] * SPLIT)
        dataset = ns219p.StockDataset(
            subset_df.head(train_split), N_FEATURES, WINDOW)
        loss = ns219p.training_loop(model, dataset, save, N_BATCH, EPOCHS)
        loss_df = loss_df.assign(**{s: loss})

    # save training loss line chart
    save = FIGURE_PATH.format("Training MSE Loss")
    if not op.isfile(save) and loss_df.size > 0:
        ax = sns.lineplot(loss_df)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average MSELoss")
        sns.despine(ax=ax)
        plt.legend(frameon=False)
        ax.get_figure().savefig(save)
        plt.clf()

    # run inference on withheld data using trained recurrent neural networks
    if not op.isfile(PREDICTION_PATH):
        ref_df = stock_df[[C_DATE, C_SYMBOL, C_CHANGE]].copy()
        ref_df[C_CHANGE] = (ref_df[C_CHANGE] * undo_z[-1]) + undo_z[0] + 1
        ref_df = pd.concat([start_df, ref_df], ignore_index=True)
        ref_df[C_CHANGE] = ref_df.groupby([C_SYMBOL])[C_CHANGE].cumprod()
        ref_df[C_SOURCE] = "Ground Truth"
        acc_df = []

        merge = [ref_df]
        for s in SUBSET:
            load = MODEL_PATH.format(s)
            seed_df = stock_df.loc[stock_df[C_SYMBOL] == s][N_FEATURES].copy()
            start = int(seed_df.shape[0] * SPLIT)
            price = ref_df.loc[ref_df[C_SYMBOL] == s].iloc[start][C_CHANGE]
            seed = seed_df.iloc[start - WINDOW:].values
            origin = seed[WINDOW:, 0].copy()
            prices = ns219p.test_loop(
                model, load, seed, WINDOW, seed_df.shape[0] - start)[..., 0]
            prices = (prices * undo_z[-1]) + undo_z[0] + 1
            acc_df += [pd.DataFrame({
                C_SYMBOL: s,
                C_CHANGE: ((origin * undo_z[-1]) + undo_z[0]).copy(),
                C_PREDICTION: (prices - 1).copy()})]
            prices = np.cumprod(np.concatenate((np.array([price]), prices)))
            seed_df = ref_df.loc[ref_df[C_SYMBOL] == s].iloc[start:].copy()
            seed_df[C_CHANGE] = prices
            seed_df[C_SOURCE] = "Model Inference"
            merge += [seed_df]

        pd.concat(merge, ignore_index=True).sort_values(
            by=[C_SYMBOL, C_DATE]).rename(columns={C_CHANGE: C_CLOSE}).to_csv(
            PREDICTION_PATH, encoding="utf-8", index=False)
        acc_df = pd.concat(acc_df, ignore_index=True).sort_values(
            by=[C_SYMBOL])

        # generate and save prediction scatter plot
        save = FIGURE_PATH.format("Confusion Scatter")
        ns219g.generate_pair_scatter(
            acc_df, save, [C_CHANGE, C_PREDICTION], C_SYMBOL)

        # generate and save prediction heatmap
        save = FIGURE_PATH.format("Confusion Matrix")
        if not op.isfile(save):
            acc = (acc_df[[C_CHANGE, C_PREDICTION]].values > THRESHOLD)
            acc = confusion_matrix(acc[:, 0], acc[:, 1])
            ax = sns.heatmap(
                pd.DataFrame(acc, index=CLASS_LABELS, columns=CLASS_LABELS),
                cbar=True, vmin=0, cbar_kws={"label": "Stock-Days"})
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Ground Truth")
            ax.get_figure().savefig(save)
            plt.clf()

    # generate and save temporal line plots to visualize predicted data
    ref_df = pd.read_csv(PREDICTION_PATH).dropna(axis=0)
    save = FIGURE_PATH.format("Predicted Data")
    ns219g.generate_facet_line(
        ref_df, save, C_SYMBOL, C_DATE, C_CLOSE, C_SOURCE)


if __name__ == '__main__':
    __main__()
