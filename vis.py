from configs import model_config
import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


# Showing images
def show_batch(dataloader, cols = model_config['show_batch_size'], image_size=32):
    # Extract one batch
    data = next(iter(dataloader))
    fig, ax = plt.subplots(2, cols, figsize=(cols*20, 2*20))
    for index in range(cols):
        ax[0, index].imshow(data[0][0][index].view(3, image_size,image_size).permute(1 , 2 , 0), interpolation='nearest')
        ax[1, index].imshow(data[0][1][index].view(3, image_size,image_size).permute(1 , 2 , 0), interpolation='nearest')
    plt.show()
    # print(data[1].numpy().reshape(-1))

    