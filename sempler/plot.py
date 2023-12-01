# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Functions for different plotting tasks; requires additional
dependencies networkx and matplotlib.

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sempler.utils as utils


def plot_graph(W, labels=None, weights=False, block=False):
    """Plot a graph with weight matrix W."""
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    pos = nx.drawing.layout.shell_layout(G, scale=0.5)
    p = len(W)
    if labels is None:
        node_labels = dict(zip(np.arange(p), map(lambda i: "$X_{%d}$" % i, range(p))))
    else:
        node_labels = dict(zip(np.arange(p), labels))
    # Plot
    fig = plt.figure()
    params = {
        "node_color": "white",
        "edgecolors": "black",
        "node_size": 900,
        "linewidths": 1.5,
        "width": 1.5,
        "arrowsize": 20,
        "arrowstyle": "->",
        "min_target_margin": 10,
        "labels": node_labels,
    }
    nx.draw(G, pos, **params)
    # Edge weights
    if weights:
        formatted = dict((e, "%0.3f" % w) for (e, w) in utils.edge_weights(W).items())
        nx.draw_networkx_edge_labels(G, pos, formatted, font_color="red")
    fig.set_facecolor("white")
    plt.show(block=block)


def plot_matrix(A, ax=None, vmin=-3, vmax=3, formt="%0.2f", thresh=1e-16, block=False):
    """Plot a heatmap for the given matrix A.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to plot.
    ax : matplotlib.pyplot.axis, optional
        The axis to plot on; if None, create a new figure.
    vmin : float, default=-3
        The lower threshold for color saturation.
    vmax : float, default=3
        The upper threshold for color saturation.
    formt : string, default="%0.2f"
        The format with which to print the values of the matrix on top
        of the corresponding cell.
    thresh : float, default=1e-16
        Elements of the matrix which are lower than the threshold in
        absolute value are plotted as a zero (i.e. white, no text)
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(A, vmin=vmin, vmax=vmax, cmap="bwr")
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] != 0:
                ax.text(j, i, formt % A[i, j], ha="center", va="center")
    plt.show(block=block)
