from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from ctc.ctc_model import load_exp
from torch.utils.data.dataloader import DataLoader
from viz_utils import batch_predict_results
import torch
import os
from PIL import Image
import numpy as np
import shutil
import json
import torch.nn.functional as F
import copy
import matplotlib.cm as mpl_color_map

shutil.rmtree('/data/kpark53/cub_vis/vis_slice/', ignore_errors=True)
shutil.rmtree('/data/kpark53/cub_vis/vis_pp_slice/', ignore_errors=True)
os.makedirs('/data/kpark53/cub_vis/vis_slice/', exist_ok=True)
os.makedirs('/data/kpark53/cub_vis/vis_pp_slice/', exist_ok=True)


def unnorm_mnist(img):
    sd = np.array([0.229, 0.224, 0.225])
    mu = np.array([0.485, 0.456, 0.406])
    img = img.transpose(0, 2).transpose(0, 1)
    return img * sd + mu

def vis(slots_vis_raw, size, thres, loc=None, index=None):

    b = slots_vis_raw.size()[0]  # torch.Size([32, 10, 196]) b=32 bcl
    # b = min(slots_vis_raw.size()[0], 50)
    if loc is not None:
        loc1, loc2 = loc
    else:
        loc1 = "/data/kpark53/cub_vis/vis_pp_slice"

    for i in range(b):
        print(f"Processing batch {i+1}/{b}")
        slots_vis = slots_vis_raw[i]  # 196,10   LC

        slots_vis_mask = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 1.).reshape(
            slots_vis.shape[:1] + (int(size) * 2, int(size) * 2))  # torch.Size([10]),14,14
        
        slots_vis_mask_new = (slots_vis_mask > thres).float()

        slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
            slots_vis.shape[:1] + (int(size) * 2, int(size) * 2))  # torch.Size([10]),14,14

        
        #slots_vis *= slots_vis_mask
        slots_vis *= slots_vis_mask_new

        slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)  # torch.Size([10, 14, 14]) -> (10, 14, 14)
        for id, image in enumerate(slots_vis):
            image = Image.fromarray(image, mode='L').resize([224, 224], resample=Image.BILINEAR)
            
            print(f"Saving image {i}_slot_{id:d}.png")
            image.save(f'{loc1}/{i}_slot_{id:d}.png')
        print(f"Batch {i+1}/{b} processed.")


def vis_slicing(slots_vis_raw, size, thres, loc=None, index_set=None):
    if index_set is not None:
        b = len(index_set)
    else:
        b = slots_vis_raw.size()[0]
    
    if loc is not None:
        loc1 = loc
    else:
        loc1 = "/data/kpark53/cub_vis/vis_pp_slice"


    for i in range(b):
        actual_index = index_set[i] if index_set is not None else i
        print(f"Processing batch {i+1}/{b}")
        slots_vis = slots_vis_raw[actual_index]

        slots_vis_mask = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 1.).reshape(
            slots_vis.shape[:1] + (int(size) * 2, int(size) * 2))  # torch.Size([10]),14,14
        
        slots_vis_mask_new = (slots_vis_mask > thres).float()

        slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
            slots_vis.shape[:1] + (int(size) * 2, int(size) * 2))  # torch.Size([10]),14,14

        
        slots_vis *= slots_vis_mask
        slots_vis *= slots_vis_mask_new

        slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)  # torch.Size([10, 14, 14]) -> (10, 14, 14)
        for id, image in enumerate(slots_vis):
            image = Image.fromarray(image, mode='L').resize([224, 224], resample=Image.BILINEAR)
            
            print(f"Saving image {actual_index}_slot_{id:d}.png")
            image.save(f'{loc1}/{actual_index}_slot_{id:d}.png')
        print(f"Batch {i+1}/{b} processed.")

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def main():
    print("Loading model and data...")
    run_path = "/data/jhong53/projects/concept_centric_transformers/cub_reduced_slotcvit_qsa/CUB2011ReducedDataModule_expl0.0/cub_reduced_slotcvit_qsa-epochs=50-lr=1e-05-expl_lambda=0.0-attention_sparsity=0.0-seed=1.ckpt"
    model, data_module = load_exp(run_path)
    results = batch_predict_results(Trainer(gpus=1).predict(model, data_module))
    
    #vis
    spa_cpt_attn = results['spatial_concept_attn']

    #original gathering(don't touch)
    spa_cpt_attn_transposed= torch.transpose(spa_cpt_attn, -2, -1)
    all_output = torch.sum(spa_cpt_attn_transposed, dim=-1)
    


    print("Generating original+concept visualizations...")
    for j in range(20):  # num_cpt
        root = '/data/kpark53/cub_vis/' + 'vis_slice/' + "cpt" + str(j + 1) + "/"
        root_slot = '/data/kpark53/cub_vis/' + 'vis_pp_slice'
        os.makedirs(root, exist_ok=True)
        selected = all_output[:, j]  # (10000,)
        ids = np.argsort(-selected, axis=0)
        idx = ids[:30]  # top_samples

        # Call vis_slicing with index_set=idx / Change threshold as you like [0,1]
        vis_slicing(spa_cpt_attn_transposed, size=7, thres=0.88, loc=root_slot, index_set=idx)

        for i in range(len(idx)):
            print(f"Processing image {i+1}/{len(idx)} for concept {j+1}/{50}")
            index = results['idx'][idx[i]].item()
            img_orl = data_module.cub_test[index][0]
            img_orl = Image.fromarray(np.uint8(unnorm_mnist(img_orl) * 255)).convert("RGBA")
            img_orl.save(root + f'/origin_{idx[i]}.png')
            slot_image_path = f'{root_slot}/{idx[i]}_slot_{j}.png'  # Update the path format
            slot_image = np.array(Image.open(slot_image_path), dtype=np.uint8)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
            heatmap_on_image.save(root + f'mask_{idx[i]}.png')
            print(f"Processed image {i+1}/{len(idx)} for concept {j+1}/{50}")
    print("Finished generating original+concept visualizations")



if __name__ == '__main__':
    main()