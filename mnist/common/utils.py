from torchvision import datasets, transforms


def get_train_test_dataset():
    data_dir = 'data/'
    apply_transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(data_dir,
                                   train=True,
                                   download=True,
                                   transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir,
                                  train=False,
                                  download=True,
                                  transform=apply_transform)

    return train_dataset, test_dataset
