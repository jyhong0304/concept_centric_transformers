from pytorch_lightning import Trainer
from ctc import load_exp
from viz_utils import batch_predict_results
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image


def load_attributes(root='/data/Datasets/'):
    # Load list of attributes
    attr_list = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'attributes.txt'),
                            sep=' ', names=['attr_id', 'def'])
    attr_list = np.array(attr_list['def'].to_list())
    return attr_list


def unnorm_cub(img):
    sd = np.array([0.229, 0.224, 0.225])
    mu = np.array([0.485, 0.456, 0.406])
    img = img.transpose(0, 2).transpose(0, 1)
    return img * sd + mu


def plot_cub_gt(sample, ind, save=False, save_path=None):
    """
        plot_cub(data_module.cub_test[2])
    """
    img, expl, spatial_expl, label = sample

    im = Image.fromarray(np.uint8(unnorm_cub(img) * 255)).convert("RGBA")

    n_patch = int(np.sqrt(spatial_expl.shape[0]))
    patch_idx = ~torch.isnan(spatial_expl[:, 0])
    patches = np.zeros(n_patch ** 2) + 0.3
    patches[patch_idx] = 1.0
    patches = patches.reshape(n_patch, n_patch)

    im_p = Image.fromarray(np.uint8(patches * 255)).convert("L")
    im_p = im_p.resize(im.size, Image.ANTIALIAS)

    im.putalpha(im_p)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    if save:
        plt.savefig(f'{save_path}/gt_{ind}.png')
    else:
        plt.show()


def plot_cub_expl(results, ind, data_module, data_root,
                  active_patch_th=0.6,
                  active_attr_th=0.3,
                  nonspatial_expl_th=0.2,
                  model_name='cvit',
                  save=False,
                  save_path=None,
                  ):
    idx = results['idx'][ind].item()

    img = data_module.cub_test[idx][0]
    im = Image.fromarray(np.uint8(unnorm_cub(img) * 255)).convert("RGBA")

    # Prediction
    pred = results['preds'][ind].item()
    prediction = data_module.cub_test.class_names[pred].split('/')[0][4:]

    # Spatial attention
    attn = results['spatial_concept_attn'][ind]
    n_patch = int(np.sqrt(attn.shape[0]))
    # Get most active patches
    patch_idx = attn.max(axis=1)[0] > active_patch_th
    if torch.count_nonzero(patch_idx) == 0:
        print(f'{ind} has zero-patch')
        return

    patches = np.zeros(n_patch ** 2) + 0.4
    patches[patch_idx] = 1.0
    patches = patches.reshape(n_patch, n_patch)

    # Get corresponding most active attributes
    attr_idx = attn[patch_idx, :].max(axis=0)[0] > active_attr_th
    attr_ind = np.nonzero(attr_idx)

    attr_list = load_attributes(root=data_root)
    attributes = attr_list[np.array(data_module.cub_test.spatial_attributes_pos)[attr_ind] - 1]

    # Nonspatial explanation
    expl = results['concept_attn'][ind]
    # print(expl)
    expl_idx = expl > nonspatial_expl_th
    nonspatial_attributes = attr_list[np.array(data_module.cub_test.non_spatial_attributes_pos)[expl_idx] - 1]

    # Plot
    im_p = Image.fromarray(np.uint8(patches * 255)).convert("L")
    im_p = im_p.resize(im.size, Image.ANTIALIAS)

    im.putalpha(im_p)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    if save:
        plt.savefig(f'{save_path}/{model_name}_{ind}.png')
    else:
        plt.show()

    correct = results['correct'][ind].item()
    correct_wrong = ['wrong', 'correct'][correct]
    if not correct:
        gt = data_module.cub_test[idx][3]
        gt = data_module.cub_test.class_names[gt].split('/')[0][4:]
        correct_wrong += f', gt is {gt}'

    result = ''
    result += f'Image index:{ind} - Prediction: {prediction} ({correct_wrong})\n'
    # print(f'Image index:{ind} - Prediction: {prediction} ({correct_wrong})')

    # print(' Spatial explanations:')
    result += ' Spatial explanations:\n'
    if isinstance(attributes, str):
        attributes = [[attributes]]
    for t in attributes:
        # print(f'   - {t[0]}')
        result += f'   - {t[0]}\n'

    # print(' Global explanations:')
    result += ' Global explanations:\n'
    for t in nonspatial_attributes:
        # print(f'   - {t}')
        result += f'   - {t}\n'

    if save:
        with open(f'{save_path}/{model_name}_{ind}.txt', 'w') as f:
            f.write(result)
    else:
        print(result)


NAME_CKPT = 'YOUR_MODEL.ckpt'
model, data_module = load_exp(NAME_CKPT)
results = batch_predict_results(Trainer(gpus=1).predict(model, data_module))
