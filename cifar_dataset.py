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


# class CustomCIFAR(Dataset):
#     def __init__(self, transform, train):
#         self.transform = transform
#         self.dataset = datasets.CIFAR10(root=root_path, train=train, download=True)

#         self.classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck','unlabelled']

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         img, label=self.dataset[index]
#         if self.transform is not None:
#             img0 = self.transform(img)
#             img1 = self.transform(img)
        

#         return img0, img1, label

# contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                           transforms.RandomResizedCrop(size=32),
#                                           transforms.RandomApply([
#                                               transforms.ColorJitter(brightness=0.9,
#                                                                      contrast=0.9,
#                                                                      saturation=0.9,
#                                                                      hue=0.4)
#                                           ], p=0.8),
#                                           transforms.RandomGrayscale(p=0.2),
#                                           transforms.GaussianBlur(kernel_size=9),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize((0.5,), (0.5,))
#                                          ])


class MyTransform():
    def __init__(self):
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.4),
                                          transforms.RandomResizedCrop(size=32),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.9,
                                                                     contrast=0.9,
                                                                     saturation=0.9,
                                                                     hue=0.4)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

transform = MyTransform()

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_val_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Initialize the network
siamese_dataset = CIFAR10(root=root_path, train=True, transform=transform, download=True)
siamese_val_dataset = CIFAR10(root=root_path, train=True, transform=train_val_transform, download=True)
siamese_testset = CIFAR10(root=root_path, train=False, transform=val_transform, download=True)

print(len(siamese_testset))
print(len(siamese_dataset))


# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=0, pin_memory=True, batch_size=model_config["batch_size"])
# Load the training without training transforms dataset
train_val_dataloader = DataLoader(siamese_val_dataset, shuffle=False, num_workers=0, pin_memory=True, batch_size=model_config["batch_size"])
# Load the testing dataset
test_dataloader = DataLoader(siamese_testset, shuffle=False, num_workers=0, pin_memory=True, batch_size=model_config["batch_size"])


vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=model_config["show_batch_size"])


train_features = next(iter(train_dataloader))
print(len(train_dataloader))
print(len(train_features[0]), len(train_features[1]))

print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0][0].shape, train_features[0][1].shape, train_features[1].shape,))

train_features = next(iter(train_val_dataloader))
print(len(train_features[0]), len(train_features[1]))
print(len(train_val_dataloader))
print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0][0].shape, train_features[0][1].shape, train_features[1].shape,))

test_features = next(iter(train_dataloader))
print(len(train_features[0]), len(train_features[1]))
print(len(test_dataloader))
print("Test Images 1 Shape: {}\nTest Images 2 Shape: {}\nTest Data Labels Shape: {}".format(test_features[0][0].shape, test_features[0][1].shape, test_features[1].shape,))


if model_config["show_batch"]:
    show_batch(vis_dataloader)

del siamese_dataset
del siamese_val_dataset
del siamese_testset

