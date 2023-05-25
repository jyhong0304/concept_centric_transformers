import torch
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt

from ctc import load_exp
from viz_utils import batch_predict_results, plot_explanation


def plot_prediction(idx):
    """Plots prediction, concept attention scores and ground truth
        explanationfor correct predictions
    """
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1, -1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1, -1), ax3)
    ax3.set_title('concept attention scores')

    return fig


def plot_wrong_prediction(num):
    """Plots prediction, concept attention scores and ground truth
        explanationfor incorrect predictions
    """
    errors_ind = torch.nonzero(results['correct'] == 0)
    # print(errors_ind)

    idx = errors_ind[num].item()
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1, -1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1, -1), ax3)
    ax3.set_title('concept attention scores')

    return fig


# Load checkpoint of trained model
NAME_CKPT = 'YOUR_MODEL.ckpt'
model, data_module = load_exp(NAME_CKPT)
results = batch_predict_results(Trainer(gpus=1).predict(model, data_module))
