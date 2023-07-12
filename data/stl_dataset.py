import torchvision
from torchvision.datasets import STL10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PIL import ImageFilter, ImageOps
import random
from configs import model_config


root_path = "D:\Ai\Projects\self-supervised-learning\data"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class STL10Pair(Dataset):
    def __init__(self, transform, train):
        self.transform = transform
        self.dataset = STL10(root=root_path, split="unlabeled" if train else "test", download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label=self.dataset[index]
        if self.transform is not None:
            img0 = self.transform(img)
            img1 = self.transform(img)
        

        return img0, img1, label



train_transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize
    # random_mask(output_size=96, mask_size=8, p=0.8)
    ])

test_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    normalize])


train_val_transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


# data prepare

def get_stl_data(seed_worker, g):
    train_data = STL10Pair(train=True, transform=train_transform)
    train_dataloader = DataLoader(train_data, batch_size=model_config["batch_size"], shuffle=True, num_workers=0, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g)

    train_val_data= STL10(root=root_path, split="train", transform=test_transform, download=True)
    train_val_dataloader  = DataLoader(train_val_data, batch_size=model_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    test_data = STL10Pair(train=False, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=model_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    
    vis_dataloader = DataLoader(train_data, shuffle=True, num_workers=0, batch_size=model_config["show_batch_size"], worker_init_fn=seed_worker, generator=g)

    return train_dataloader, train_val_dataloader, test_dataloader, vis_dataloader

# train_features = next(iter(train_dataloader))
# print(len(train_dataloader))
# print(len(train_features[0]), len(train_features[1]), len(train_features[2]))

# print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0][0].shape, train_features[0][1].shape, train_features[1].shape,))

# train_features = next(iter(train_val_dataloader))
# print(len(train_features[0]), len(train_features[1]))
# print(len(train_val_dataloader))
# print("Train Images 1 Shape: {}\nTrain Images 2 Shape: {}\nTrain Data Labels Shape: {}".format(train_features[0][0].shape, train_features[0][1].shape, train_features[1].shape,))

# test_features = next(iter(test_dataloader))
# print(len(train_features[0]), len(train_features[1]))
# print(len(test_dataloader))
# print("Test Images 1 Shape: {}\nTest Images 2 Shape: {}\nTest Data Labels Shape: {}".format(test_features[0][0].shape, test_features[0][1].shape, test_features[1].shape,))


# del siamese_dataset
# del siamese_val_dataset
# del siamese_testset

