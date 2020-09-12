
# dataset.py

from torchvision import datasets

def mnist():
  train_transforms, test_transforms = mnist_transforms()
  train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
  test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
  return train, test
