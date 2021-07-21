import io
import os
import pickle

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, KMNIST, STL10, SVHN, FashionMNIST, ImageFolder

# from meta_imagenet.lib.datasets.DownsampledImageNet import ImageNet32


NUM_CLASSES = {
    "aircraft": 100,
    "cifar_100": 100,
    "cub": 200,
    "cub_256": 200,
    "fashion_mnist": 10,
    "stanford_cars": 196,
    "stanford_dogs": 120,
    "stanford_dogs_256": 120,
    "stl10": 10,
    "svhn": 10,
    "tiny_imagenet": 200,
    "tiny_imagenet_split_1": 100,
    "tiny_imagenet_split_2": 100,
    "mini_imagenet": 200,
    "dtd": 47,
    "kmnist": 10,
    "vgg_flower_102": 102,
    "stanford_40": 40,
    "quickdraw": 345,
    "traffic_sign": 43,
    "vgg_pets": 37,
    "food_101": 101,
    "birdsnap": 500,
    "sun397": 397,
    "stanford_40_actions": 40,
    "fungi": 1394,
    "fruit_360": 131,
    "deepweeds": 9,
    "voc2007": 20,
    "imagenet_8split_0": 113,
    "imagenet_8split_1": 131,
    "imagenet_8split_2": 121,
    "imagenet_8split_3": 121,
    "imagenet_8split_4": 120,
    "imagenet_8split_5": 95,
    "imagenet_8split_6": 155,
    "imagenet_8split_7": 144,
    "imagenet_2split_0": 500,
    "imagenet_2split_1": 500,
    "chest_xray": 2,
    "mvtec": 88,
    "cifar_10": 10,
}
DS_MEAN = {
    "birdsnap": [0.488, 0.502, 0.457],
    "sun397": [0.476, 0.460, 0.425],
    "dtd": [0.530, 0.473, 0.425],
    "vgg_pets": [0.478, 0.446, 0.396],
    "stanford_40_actions": [0.467, 0.441, 0.402],
    "fungi": [0.439, 0.413, 0.339],
    "fruit_360": [0.684, 0.579, 0.504],
    "deepweeds": [0.378, 0.390, 0.380],
    "aircraft": [0.482, 0.512, 0.536],
    "stanford_cars": [0.471, 0.46, 0.455],
    "stanford_dogs": [0.477, 0.452, 0.391],
    "stanford_dogs_256": [0.477, 0.452, 0.391],
    "vgg_flower_102": [0.433, 0.382, 0.297],
    "food_101": [0.545, 0.444, 0.344],
    "quickdraw": [0.169],
    "fashion_mnist": [0.286],
    "cub": [0.486, 0.5, 0.433],
    "cub_256": [0.486, 0.5, 0.433],
    "mvtec": [0.433, 0.404, 0.395],
    "chest_xray": [0.482],
    "voc2007": [0.449, 0.422, 0.388],
    "traffic_sign": [0.340, 0.312, 0.321],
    "cifar_100": [0.508, 0.487, 0.441],
    "cifar_10": [0.491, 0.482, 0.447],
    "svhn": [0.438, 0.444, 0.473],
    "tiny_imagenet_split_1": [0.474, 0.446, 0.391],
    "tiny_imagenet_split_2": [0.486, 0.450, 0.404],
}
DS_STD = {
    "birdsnap": [0.220, 0.218, 0.259],
    "sun397": [0.256, 0.254, 0.278],
    "dtd": [0.261, 0.251, 0.259],
    "vgg_pets": [0.262, 0.257, 0.264],
    "stanford_40_actions": [0.271, 0.264, 0.274],
    "fungi": [0.248, 0.234, 0.240],
    "fruit_360": [0.303, 0.360, 0.391],
    "deepweeds": [0.205, 0.203, 0.201],
    "aircraft": [0.217, 0.211, 0.243],
    "stanford_cars": [0.288, 0.287, 0.296],
    "stanford_dogs": [0.256, 0.251, 0.255],
    "stanford_dogs_256": [0.256, 0.251, 0.255],
    "vgg_flower_102": [0.291, 0.243, 0.27],
    "food_101": [0.269, 0.271, 0.275],
    "quickdraw": [0.296],
    "fashion_mnist": [0.339],
    "cub": [0.226, 0.221, 0.26],
    "cub_256": [0.226, 0.221, 0.26],
    "mvtec": [0.258, 0.255, 0.252],
    "chest_xray": [0.236],
    "voc2007": [0.266, 0.262, 0.274],
    "traffic_sign": [0.276, 0.265, 0.271],
    "cifar_100": [0.262, 0.251, 0.271],
    "cifar_10": [0.241, 0.238, 0.256],
    "svhn": [0.196, 0.199, 0.196],
    "tiny_imagenet_split_1": [0.270, 0.261, 0.271],
    "tiny_imagenet_split_2": [0.284, 0.277, 0.292],
}


def get_dataset(data: str, train: bool, transform=None, target_transform=None) -> Dataset:
    if data == "cifar_10":
        return CIFAR10("data/cifar_10", train, transform, target_transform, download=True)
    elif data == "cifar_100":
        return CIFAR100("data/cifar_100", train, transform, target_transform, download=True)
    elif data == "fashion_mnist":
        return FashionMNIST("data/fashion_mnist", train, transform, target_transform, download=True)
    elif data == "stl10":
        return STL10(
            "data/stl10", "train" if train else "test", None, transform, target_transform, download=True
        )
    elif data == "svhn":
        return SVHN("data/svhn", "train" if train else "test", transform, target_transform, download=True)
    elif data == "tiny_imagenet":
        return TinyImageNet(train, transform, target_transform)
    elif data == "tiny_imagenet_split_1":
        ds = TinyImageNet(train, transform, target_transform)
        indices = [i for i, l in enumerate(ds.labels) if l < 100]
        return Subset(ds, indices)
    elif data == "tiny_imagenet_split_2":
        if target_transform is None:
            target_transform = lambda y: y - 100
        else:
            target_transform = transforms.Compose([target_transform, lambda y: y - 100])
        ds = TinyImageNet(train, transform, target_transform)
        indices = [i for i, l in enumerate(ds.labels) if l >= 100]
        return Subset(ds, indices)
    elif data == "kmnist":
        return KMNIST("data/kmnist", train, transform, target_transform, download=True)
    elif data == "quickdraw":
        return NumpyDataset(
            image_path="data/quickdraw/{}_images.npy".format("train" if train else "test"),
            label_path="data/quickdraw/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )
    elif data == "traffic_sign":
        return NumpyDataset(
            image_path="data/traffic_sign/{}_images.npy".format("train" if train else "test"),
            label_path="data/traffic_sign/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )
    else:
        path = f"data/{data}/{'train' if train else 'test'}.lmdb"
        if not os.path.exists(path):
            raise NotImplementedError
        return ImageFolderLMDB(path, transform, target_transform)


class ImageNetEpisodic(Dataset):
    def __init__(self, db_path, indices, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path, subdir=os.path.isdir(db_path), readonly=True, lock=False, readahead=False, meminit=False
        )
        self.indices = indices
        with self.env.begin(write=False) as txn:
            # self.length = pickle.loads(txn.get(b"__len__"))
            # self.keys = pickle.loads(txn.get(b"__keys__"))
            self.class_indices = pickle.loads(txn.get(b"__cls_ind__"))
            self.update_indices(indices)

        self.transform = transform
        self.target_transform = target_transform

    def update_indices(self, indices):
        self.keys = [k for c in indices for k in self.class_indices[c]]
        self.length = len(self.keys)

    def __getitem__(self, index):
        img, target = None, None
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(f"{self.keys[index]}".encode("ascii"))
        imgbuf, target = pickle.loads(byteflow)
        target = self.indices.index(target)

        # load image
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


class ImageNetSplit(ImageFolder):
    def __init__(self, train, split_num, transform, target_transform):
        self.split_num = split_num
        super().__init__(
            f"data/imagenet1k/raw-data/{'train' if train else 'val'}/", transform, target_transform
        )

    def _find_classes(self, dir):
        with open(f"data/imagenet_split/split_{self.split_num}.txt", "r") as f:
            classes = [l.strip() for l in f.readlines()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ImageNet2Split(ImageFolder):
    def __init__(self, train, split_num, transform, target_transform):
        self.split_num = split_num
        super().__init__(
            f"data/imagenet1k/raw-data/{'train' if train else 'val'}/", transform, target_transform
        )

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        if self.split_num == 0:
            classes = classes[:500]
        else:
            classes = classes[500:]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path, subdir=os.path.isdir(db_path), readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        imgbuf, target = pickle.loads(byteflow)

        # load image
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


class NumpyDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(image_path)
        self.labels = np.load(label_path)
        self.length = len(self.images)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.length


class TinyImageNet(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="/data/tiny_imagenet/{}_images.npy".format("train" if train else "valid"),
            label_path="/data/tiny_imagenet/{}_labels.npy".format("train" if train else "valid"),
            transform=transform,
            target_transform=target_transform,
        )


class MiniImageNet(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="/data/mini_imagenet/{}_images.npy".format("train" if train else "valid"),
            label_path="/data/mini_imagenet/{}_labels.npy".format("train" if train else "valid"),
            transform=transform,
            target_transform=target_transform,
        )


class CUB(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="/data/CUB_200_2011/84_npy/{}_images.npy".format("train" if train else "test"),
            label_path="/data/CUB_200_2011/84_npy/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class Aircraft(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="/data/aircraft/{}_images.npy".format("train" if train else "test"),
            label_path="/data/aircraft/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class StanfordCars(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="/data/stanford_cars/{}_images.npy".format("train" if train else "test"),
            label_path="/data/stanford_cars/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class StanfordDogs(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="/data/stanford_dogs/{}_images.npy".format("train" if train else "test"),
            label_path="/data/stanford_dogs/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )
