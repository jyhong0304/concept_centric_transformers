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

# Define constants for directory paths and threshold
DATA_DIR = './imagenet_vis/'
VIS_DIR = '/vis_20/'  # Change this as needed
VIS_PP_DIR = '/vis_pp_20/'  # Change this as needed
THRESHOLD = 0.5 # Change this as needed


def unnorm_mnist(img):
    sd = np.array([0.229, 0.224, 0.225])
    mu = np.array([0.485, 0.456, 0.406])
    img = img.transpose(0, 2).transpose(0, 1)
    return img * sd + mu


def vis(slots_vis_raw, size, loc=None, index=None):
    global THRESHOLD  # Use the global threshold variable
    b = slots_vis_raw.size()[0]  

    for i in range(b):
        print(f"Processing batch {i+1}/{b}")
        slots_vis = slots_vis_raw[i]  

        slots_vis_mask = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 1.).reshape(
            slots_vis.shape[:1] + (int(size) * 2, int(size) * 2)) #slot mask normalization [0,1]

        slots_vis_mask_new = (slots_vis_mask > THRESHOLD).float() #Take the only slots that are above the threshold (threshold is user-defined)

        slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
            slots_vis.shape[:1] + (int(size) * 2, int(size) * 2)) #slot mask normalization [0,255] for visualization

        slots_vis *= slots_vis_mask_new #Only significant slots are kept

        slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
        for id, image in enumerate(slots_vis):
            image = Image.fromarray(image, mode='L').resize([224, 224], resample=Image.BILINEAR) #slot mask resizing to 224x224, which is the size of original image from dataset

            print(f"Saving image {i}_slot_{id:d}.png")
            image.save(os.path.join(loc, f'{i}_slot_{id:d}.png'))
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


def main(data_dir, vis_dir, vis_pp_dir, threshold):
    global DATA_DIR, VIS_DIR, VIS_PP_DIR, THRESHOLD  # Access constants

    DATA_DIR = data_dir  # Set data directory(ckpt)
    VIS_DIR = vis_dir  # Set original image directory
    VIS_PP_DIR = vis_pp_dir  # Set slot mask directory
    THRESHOLD = threshold  # Set threshold

    print("Loading model and data...")
    run_path = os.path.join(DATA_DIR, "your_model_path_here.ckpt")  # Change this as needed
    model, data_module = load_exp(run_path)
    results = batch_predict_results(Trainer().predict(model, data_module))
    

    spa_cpt_attn = results['spatial_concept_attn'] # vis attn mask from ckpt dataloader

    # Similar images gathered together based on the C of concept attention mask
    spa_cpt_attn_transposed= torch.transpose(spa_cpt_attn, -2, -1)
    all_output = torch.sum(spa_cpt_attn_transposed, dim=-1)

    
    
    print("Visualizing spatial concept attention...")
    vis(spa_cpt_attn_transposed, size=7)
    print("Finished spatial concept attention visualization")

    #Select the top 20 samples from the test dataset for each concept(10), and then generate visualizations that combine the original with the slot make in focus
    print("Generating original+concept visualizations...")
    for j in range(10):  # number of concepts. Here is 10. Defined by user
        root = DATA_DIR + VIS_DIR + "cpt" + str(j + 1) + "/"
        root_slot = DATA_DIR + VIS_PP_DIR
        os.makedirs(root, exist_ok=True)
        selected = all_output[:, j]  
        ids = np.argsort(-selected, axis=0) #sort in descending order
        idx = ids[:20]  # pick top_20_samples
        for i in range(len(idx)):
            print(f"Processing image {i+1}/{len(idx)} for concept {j+1}/{50}")
            index = results['idx'][idx[i]].item() #index of image in test dataset
            img_orl = data_module.imagenet_test[index][0] #loading original image from test dataset
            img_orl = Image.fromarray(np.uint8(unnorm_mnist(img_orl) * 255)).convert("RGBA") #original image from test dataset
            img_orl.save(root + f'/origin_{idx[i]}.png')  #save original image
            slot_image_path = f'{root_slot}/{idx[i]}_slot_{j}.png'  #slot mask image path
            slot_image = np.array(Image.open(slot_image_path), dtype=np.uint8) #load slot mask image
            _, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet') #put slot mask on original image
            heatmap_on_image.save(root + f'mask_{idx[i]}.png') # save slot mask on original image
            print(f"Processed image {i+1}/{len(idx)} for concept {j+1}/{50}")
    print("Finished generating original+concept visualizations")



if __name__ == '__main__':
    data_dir = './imagenet_vis/'
    vis_dir = '/vis_20/' 
    vis_pp_dir = '/vis_pp_20/'  
    threshold = 0.5  # change as needed
    main(data_dir, vis_dir, vis_pp_dir, threshold)
