#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 05 09:00:00 2024
@author: ike
"""


import numpy as np
import seaborn as sns
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


"""
Helper functions to generate and save graphs.
"""
# style of seaborn backend for graphs
sns.set_theme(style="ticks")


def generate_pair_line(data, save, x_vars, y_vars, hue):
    if op.isfile(save):
        return

    f = sns.PairGrid(
        data, hue=hue, x_vars=x_vars, y_vars=y_vars, diag_sharey=True,
        aspect=len(y_vars))
    f.map(sns.lineplot)
    f.add_legend()
    f.savefig(save)
    plt.clf()


def generate_facet_line(data, save, r_var, x_var, y_var, hue):
    if op.isfile(save):
        return

    f = sns.FacetGrid(
        data=data, row=r_var, hue=hue, aspect=data[r_var].nunique(),
        sharey=False)
    f.map(sns.lineplot, x_var, y_var, n_boot=0, errorbar=None)
    f.set(xlim=(data[x_var].min(), data[x_var].max()))
    for ax in f.axes.flat:
        ax.xaxis.set_major_locator(tic.MaxNLocator(nbins=12))

    f.add_legend()
    f.savefig(save)
    plt.clf()


def generate_pair_scatter(data, save, i_vars, hue, style=None):
    if op.isfile(save):
        return

    f = sns.PairGrid(
        data, hue=hue, vars=i_vars, diag_sharey=False, corner=True)
    f.map_lower(sns.scatterplot, style=None if style is None else data[style])
    f.map_diag(sns.kdeplot, fill=True, alpha=0.75)
    f.add_legend()
    f.savefig(save, dpi=200)
    plt.clf()


def generate_3d_plot(data, save, i_vars, hue):
    if op.isfile(save):
        return

    c_dict = {l: i for i, l in enumerate(np.unique(data[hue]))}
    groups = data[hue].apply(lambda x: c_dict[x])
    cmap = ListedColormap(
        sns.color_palette(n_colors=10).as_hex()[:len(c_dict)])
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    sc = ax.scatter(
        *[data[c] for c in i_vars], s=20, c=groups, cmap=cmap, alpha=1)
    ax.set_xlabel(i_vars[0])
    ax.set_ylabel(i_vars[1])
    ax.set_zlabel(i_vars[2])
    plt.legend(
        handles=sc.legend_elements()[0], labels=c_dict.keys(),
        bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig(save, bbox_inches='tight')
    plt.clf()
