from configs import model_config
import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os


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
    # fig, ax = plt.subplots(2, cols, figsize=(cols*20, 2*20))
    # for index in range(cols):
    #     ax[0, index].imshow(data[0][index].view(3, image_size,image_size).permute(1 , 2 , 0), interpolation='nearest')
    #     ax[1, index].imshow(data[0][index].view(3, image_size,image_size).permute(1 , 2 , 0), interpolation='nearest')
    # plt.show()
    # print(data[1].numpy().reshape(-1))

    