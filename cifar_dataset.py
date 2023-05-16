import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from configs import model_config
from PIL import Image
from PIL import ImageFilter, ImageOps
from vis import show_batch
import random


root_path = "D:\Ai\Projects\self-supervised-learning\data"

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# data prepare
train_data = CIFAR10Pair(root=root_path, train=True, transform=train_transform, download=True)
train_dataloader = DataLoader(train_data, batch_size=model_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

train_val_data= CIFAR10(root=root_path, train=True, transform=test_transform, download=True)
train_val_dataloader  = DataLoader(train_val_data, batch_size=model_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

test_data = CIFAR10(root=root_path, train=False, transform=test_transform, download=True)
test_dataloader = DataLoader(test_data, batch_size=model_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)


vis_dataloader = DataLoader(test_data, shuffle=True, num_workers=0, batch_size=model_config["show_batch_size"])


train_features = next(iter(train_dataloader))
print(len(train_dataloader))
print(len(train_features[0]), len(train_features[1]))

print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0][0].shape, train_features[0][1].shape, train_features[1].shape,))

train_features = next(iter(train_val_dataloader))
print(len(train_features[0]), len(train_features[1]))
print(len(train_val_dataloader))
print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0][0].shape, train_features[0][1].shape, train_features[1].shape,))

test_features = next(iter(test_dataloader))
print(len(train_features[0]), len(train_features[1]))
print(len(test_dataloader))
print("Test Images 1 Shape: {}\nTest Images 2 Shape: {}\nTest Data Labels Shape: {}".format(test_features[0][0].shape, test_features[0][1].shape, test_features[1].shape,))


if model_config["show_batch"]:
    show_batch(vis_dataloader)

# del siamese_dataset
# del siamese_val_dataset
# del siamese_testset

