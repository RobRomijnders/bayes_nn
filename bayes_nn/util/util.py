import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import torch
import os


def to_tensor(im, lbl, CUDA=True):
    """
    Set the numpy data to PyTorch datatypes
    :param im:
    :param lbl:
    :return:
    """
    im = torch.FloatTensor(im)
    lbl = torch.LongTensor(lbl)
    if CUDA:
        im, lbl = im.cuda(), lbl.cuda()
    return im, lbl


def get_color(pred_class, labels):
    """
    Function to color the barplots. Defaults to blue, but it colors
    the prediction green/red if it's correct/incorrect
    :param pred_class:
    :param labels:
    :return:
    """
    num_classes = 10  # Hard coded for the 10 MNIST labels
    target_class = np.argmax(labels)

    colors = ['b'] * num_classes
    if pred_class == target_class:
        color = 'g'
    else:
        color = 'r'

    colors[pred_class] = color
    return colors


def reduce_entropy(X, axis=-1):
    return -1 * np.sum(X * np.log(X+1E-12), axis=axis)


def calc_risk(preds, labels=None, weights=None):
    """
    Calculates the parameters we can possibly use to examine risk of a neural net

    In case of Langevin dynamics, use weigths to implement eqn 11 in
    https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
    :param preds:
    :param labels:
    :param weights: weights for the weighted MC estimate
    :return:
    """
    if isinstance(preds, list):
        preds = np.stack(preds)
    # preds in shape [num_runs, num_batch, num_classes]
    num_runs, num_batch = preds.shape[:2]

    if weights is not None:
        assert weights.shape[0] == num_runs
        weights *= num_runs/np.sum(weights)  # Make the weights sum to num_runs
        preds *= np.expand_dims(np.expand_dims(weights, axis=1), axis=2)

    ave_preds = np.mean(preds, 0)
    pred_class = np.argmax(ave_preds, 1)

    # entropy = np.mean(-1 * np.sum(preds * np.log(preds+1E-12), axis=-1), axis=0)
    entropy = reduce_entropy(ave_preds, -1)  # entropy of the posterior predictive
    entropy_exp = np.mean(reduce_entropy(preds, -1), axis=0)  # Expected entropy of the predictive under the parameter posterior
    mutual_info = entropy - entropy_exp  # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf
    variance = np.std(preds[:, range(num_batch), pred_class], 0)
    ave_softmax = np.mean(preds[:, range(num_batch), pred_class], 0)
    if labels is not None:
        correct = np.equal(pred_class, labels)
    else:
        correct = None
    return entropy, mutual_info, variance, ave_softmax, correct


def plot_preds(preds, batch):
    """
    Lots of matplotlib magic to plot the predictions
    :param preds:
    :param batch: tuple of images, labels
    :return:
    """
    images, labels = batch
    if isinstance(preds, list):
        preds = np.stack(preds)

    num_samples, num_batch, num_classes = preds.shape

    ave_preds = np.mean(preds, 0)
    pred_class = np.argmax(ave_preds, 1)

    entropy, variance, _, _ = calc_risk(preds)

    # Do all the plotting

    for n in range(num_batch):
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        half = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        colors = get_color(pred_class[n], labels[n])
        for num_sample in range(half._ncols * half._nrows):
            ax = plt.Subplot(fig, half[num_sample])
            ax.bar(range(10), preds[num_sample, n], color=colors)
            ax.set_ylim(0, np.max(preds))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

        half = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, half[0])
        ax.imshow(np.squeeze(images[n]))
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, half[1])
        ax.bar(range(10), ave_preds[n], color=colors)
        ax.set_ylim(0, np.max(preds))
        ax.set_xticks([])
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, half[2])
        t = ax.text(0.5, 0.5, 'Entropy %7.3f \n Std %7.3f' % (entropy[n], variance[n]))
        t.set_ha('center')
        fig.add_subplot(ax)

        # fig.show()
        plt.savefig('im/plot%i.png' % n)


def maybe_make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
