import torch
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt
from torchvision import transforms

from ctc import load_exp
from viz_utils import batch_predict_results, remove_spines
import numpy as np
from torchvision.datasets import CIFAR100
import os

predict_labs = {
    0: "aquatic_mammals",
    1: "fish",
    2: "flowers",
    3: "food_containers",
    4: "fruit_and_vegetables",
    5: "household_electrical_devices",
    6: "household_furniture",
    7: "insects",
    8: "large_carnivores",
    9: "large_man-made_outdoor_things",
    10: "large_natural_outdoor_scenes",
    11: "large_omnivores_and_herbivores",
    12: "medium_mammals",
    13: "non-insect_invertebrates",
    14: "people",
    15: "reptiles",
    16: "small_mammals",
    17: "trees",
    18: "vehicles_1",
    19: "vehicles_2",
}

correct_labs = {0: 'wrong', 1: 'correct'}

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

DATA_PATH = 'YOUR_DATA_PATH'
cifar100_test = CIFAR100(train=False, root=DATA_PATH, transform=test_transform)


def plot_prediction(idx):
    """Plots prediction, concept attention scores and ground truth
        explanation for correct predictions
    """
    # img = data_module.cifar100_test[idx][0]
    img = cifar100_test[idx][0]

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(transforms.ToPILImage()(img))
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(10, -1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(10, -1), ax3)
    ax3.set_title('concept attention scores')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    return fig


def plot_wrong_prediction(num):
    """Plots prediction, concept attention scores and ground truth
        explanationfor incorrect predictions
    """
    errors_ind = torch.nonzero(results['correct'] == 0)
    # print(errors_ind)

    idx = errors_ind[num].item()
    img = cifar100_test[idx][0]

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(transforms.ToPILImage()(img))
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(10, -1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(10, -1), ax3)
    ax3.set_title('concept attention scores')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    return fig


def plot_explanation(raster, ax):
    ax.imshow(raster)
    # Major ticks
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, 11, 1))
    ax.set_yticklabels(np.arange(1, 11, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)
    remove_spines(ax)


def test_class_accuracy():
    size_data = results['expl'].shape[0]
    count_correct = 0.
    for i in range(size_data):
        idx_gt = torch.argmax(results['expl'][i])
        idx_pred = torch.argmax(results['concept_attn'][i])
        if idx_gt == idx_pred:
            count_correct += 1

    return count_correct / size_data


# Load checkpoint of trained model
NAME_CKPT = 'YOUR_MODEL.ckpt'
model, data_module = load_exp(NAME_CKPT)
results = batch_predict_results(Trainer().predict(model, data_module))