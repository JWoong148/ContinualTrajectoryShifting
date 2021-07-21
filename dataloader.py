from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.get_dataset import DS_MEAN, DS_STD, NUM_CLASSES, get_dataset

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


# def get_transform(
#     img_size, random_crop=False, random_horizontal_flip=False, normalize_mean=(0.5,), normalize_std=(0.5,)
# ):
#     transform_list = []
#     if random_crop:
#         transform_list.append(transforms.RandomResizedCrop((img_size, img_size)))
#     else:
#         transform_list.append(transforms.Resize((img_size, img_size)))
#     if random_horizontal_flip:
#         transform_list.append(transforms.RandomHorizontalFlip())
#     transform_list.append(transforms.ToTensor())
#     transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
#     return transforms.Compose(transform_list)


def get_transform(
    img_size, random_crop=False, random_horizontal_flip=False, normalize_mean=(0.5,), normalize_std=(0.5,)
):
    transform_list = [transforms.Resize((img_size, img_size))]
    if random_crop:
        transform_list.append(transforms.RandomCrop(img_size, padding=(4 if img_size == 32 else 8)))
    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
    return transforms.Compose(transform_list)


def get_dataloader(name, img_size, batch_size, num_workers=1, train_val_split=0.75):
    # mean = DS_MEAN[name] if name in DS_MEAN.keys() else IMAGENET_MEAN
    # std = DS_STD[name] if name in DS_STD.keys() else IMAGENET_STD
    # transform_train = get_transform(
    #     img_size, random_crop=True, random_horizontal_flip=True, normalize_mean=mean, normalize_std=std
    # )
    # transform_test = get_transform(img_size, normalize_mean=mean, normalize_std=std)
    transform_train = get_transform(img_size, random_crop=True, random_horizontal_flip=True)
    transform_test = get_transform(img_size)

    train_ds = get_dataset(name, train=True, transform=transform_train)
    lengths = [int(train_val_split * len(train_ds)), len(train_ds) - int(train_val_split * len(train_ds))]
    train_ds, val_ds = random_split(train_ds, lengths)
    test_ds = get_dataset(name, train=False, transform=transform_test)

    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, "drop_last": True}

    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=True, **kwargs)
    test_loader = DataLoader(test_ds, **kwargs)

    return train_loader, val_loader, test_loader, NUM_CLASSES[name]


# def get_dataloader(name, img_size, batch_size, num_workers=1):
#     # mean = DS_MEAN[name] if name in DS_MEAN.keys() else IMAGENET_MEAN
#     # std = DS_STD[name] if name in DS_STD.keys() else IMAGENET_STD
#     # transform_train = get_transform(
#     #     img_size, random_crop=True, random_horizontal_flip=True, normalize_mean=mean, normalize_std=std
#     # )
#     # transform_test = get_transform(img_size, normalize_mean=mean, normalize_std=std)
#     transform_train = get_transform(img_size, random_crop=True, random_horizontal_flip=True)
#     transform_test = get_transform(img_size)

#     train_ds = get_dataset(name, train=True, transform=transform_train)
#     test_ds = get_dataset(name, train=False, transform=transform_test)

#     kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, "drop_last": True}

#     train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
#     test_loader = DataLoader(test_ds, **kwargs)

#     return train_loader, test_loader, NUM_CLASSES[name]
