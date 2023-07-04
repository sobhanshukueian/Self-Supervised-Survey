from configs import model_config
import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from utils.utils import get_colors
import numpy as np
import os.path as osp

# Showing images
def show_batch(dataloader, cols = model_config['show_batch_size'], image_size=32):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # Extract one batch
    data = next(iter(dataloader))
    print(data[0].size())
    # Assuming your tensor is called 'images'
    grid1 = vutils.make_grid(data[0], nrow=model_config['show_batch_size'])  # nrow sets the number of images per row in the grid
    grid2 = vutils.make_grid(data[1], nrow=model_config['show_batch_size'])  # nrow sets the number of images per row in the grid
    fig, ax = plt.subplots(2, 1, figsize=(cols*20, 2*20))
    ax[0].imshow(grid1.permute(1, 2, 0))
    ax[1].imshow(grid2.permute(1, 2, 0))
    # plt.imshow(grid.permute(1, 2, 0))  # Permute the tensor dimensions to (H, W, C) for displaying in matplotlib
    plt.axis('off')
    plt.show()


def plot_embeddings(epoch, val_embeddings, val_labels, val_plot_size=0):
    if val_plot_size > 0:
        val_embeddings = np.array(val_embeddings[:val_plot_size])
        val_labels = np.array(val_labels[:val_plot_size])

    OUTPUT_EMBEDDING_SIZE = 10

    COLS = int(OUTPUT_EMBEDDING_SIZE / 2)
    ROWS = 1
    fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
    # fig.suptitle("Embeddings Plot", fontsize=16)
    for dim in range(0, OUTPUT_EMBEDDING_SIZE-1, 2):
        ax[int(dim/2)].set_title("Validation Samples for {} and {} dimensions".format(dim, dim+1))
        ax[int(dim/2)].scatter(val_embeddings[:, dim], val_embeddings[:, dim+1], c=get_colors(np.squeeze(val_labels)))
        
    if model_config["SAVE_PLOTS"]:
        save_plot_dir = osp.join(model_config["SAVE_DIR"], 'plots') 
        if not osp.exists(save_plot_dir):
            os.makedirs(save_plot_dir)
        plt.savefig("{}/epoch-{}-plot.png".format(save_plot_dir, epoch)) 
    if model_config["VISUALIZE_PLOTS"]:
        plt.show()
    
    plt.close('all')

    # fig, ax = plt.subplots(2, cols, figsize=(cols*20, 2*20))
    # for index in range(cols):
    #     ax[0, index].imshow(data[0][index].view(3, image_size,image_size).permute(1 , 2 , 0), interpolation='nearest')
    #     ax[1, index].imshow(data[0][index].view(3, image_size,image_size).permute(1 , 2 , 0), interpolation='nearest')
    # plt.show()
    # print(data[1].numpy().reshape(-1))

    