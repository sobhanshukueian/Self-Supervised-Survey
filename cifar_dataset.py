import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from configs import model_config

root_path = "D:\Ai\Projects\self-supervised-learning\data"
class CustomCIFAR(Dataset):
    def __init__(self, transform, train):
        self.transform = transform
        self.dataset = datasets.CIFAR10(root=root_path, train=train, download=True)

        self.classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck','unlabelled']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label=self.dataset[index]
        if self.transform is not None:
            img0 = self.transform(img)
            img1 = self.transform(img)
        

        return img0, img1, label

contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
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
val_transformation = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
# Initialize the network
siamese_dataset = CustomCIFAR(contrast_transforms, True)
siamese_val_dataset = CustomCIFAR(val_transformation, True)
siamese_testset = CustomCIFAR(val_transformation, False)

# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=0, batch_size=model_config["batch_size"])
# Load the training without training transforms dataset
train_val_dataloader = DataLoader(siamese_val_dataset, shuffle=False, num_workers=0, batch_size=model_config["batch_size"])
# Load the testing dataset
test_dataloader = DataLoader(siamese_testset, shuffle=False, num_workers=0, batch_size=model_config["batch_size"])
vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=model_config["show_batch_size"])



# train_features = next(iter(train_dataloader))
# print(len(train_dataloader))
# print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0].shape, train_features[1].shape, train_features[2].shape,))

# train_features = next(iter(train_val_dataloader))
# print(len(train_val_dataloader))
# print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0].shape, train_features[1].shape, train_features[2].shape,))

# test_features = next(iter(train_dataloader))
# print(len(test_dataloader))
# print("Test Images 1 Shape: {}\nTest Images 2 Shape: {}\nTest Data Labels Shape: {}".format(test_features[0].shape, test_features[1].shape, test_features[2].shape,))


del siamese_dataset
del siamese_val_dataset
del siamese_testset

