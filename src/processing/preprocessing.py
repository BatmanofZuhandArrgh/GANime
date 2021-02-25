import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

#Other preproc and postproc functions:
def denorm(img_tensors, stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64, figsize = (14,14)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    # ax.imshow(make_grid(images.detach()[:nmax], nrow=8).permute(1, 2, 0))