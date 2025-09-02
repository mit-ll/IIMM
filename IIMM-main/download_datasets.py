import os
import requests
import zipfile

from torchvision.datasets import (
    CIFAR100,
    DTD,
    EuroSAT,
    GTSRB,
    MNIST,
    STL10,
    SVHN
)
from wilds import get_dataset as get_wilds_dataset


def delete_file(pathname):
    try:
        os.remove(pathname)
    except FileNotFoundError:
        pass


# Create root directory
data_root = "./data"
print(f"Data will be downloaded to {data_root}.")

if not os.path.exists(data_root):
    os.makedirs(data_root)

### Download datasets ###
print("Downloading CIFAR100..")
cifar_train = CIFAR100(root=data_root, train=True, download=True)
cifar_test = CIFAR100(root=data_root, train=False, download=True)
delete_file(os.path.join(data_root, "cifar-100-python.tar.gz"))

print("Downloading DTD..")
dtd_train = DTD(root=data_root, split="train", download=True)
dtd_val = DTD(root=data_root, split="val", download=True)
dtd_test = DTD(root=data_root, split="test", download=True)
delete_file(os.path.join(data_root, "dtd", "dtd-r1.0.1.tar.gz"))

print("Downloading EuroSAT..")
eurosat = EuroSAT(root=data_root, download=True)
delete_file(os.path.join(data_root, "eurosat", "EuroSAT.zip"))

print("Downloading FMoW..")
fmow = get_wilds_dataset(dataset="fmow", download=True, root_dir=data_root)
delete_file(os.path.join(data_root, "fmow_v1.1", "archive.tar.gz"))

print("Downloading GTSRB..")
gtsrb_train = GTSRB(root=data_root, split="train", download=True)
gtsrb_test = GTSRB(root=data_root, split="test", download=True)
for item in os.listdir(os.path.join(data_root, "gtsrb")):
    if item.endswith(".zip"):
        delete_file(os.path.join(data_root, "gtsrb", item))

print("Downloading MNIST..")
mnist_train = MNIST(root=data_root, train=True, download=True)
mnist_test = MNIST(root=data_root, train=False, download=True)

print("Downloading STL10..")
stl10_train = STL10(root=data_root, split="train", download=True)
stl10_test = STL10(root=data_root, split="test", download=True)
delete_file(os.remove(os.path.join(data_root, "stl10_binary.tar.gz")))

print("Downloading SUN397..")
download_url = "https://3dvision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
sun_save_dir = os.path.join(data_root, "SUN397")
if not os.path.exists(sun_save_dir):
    try:
        response = requests.get("https://3dvision.princeton.edu/projects/2010/SUN/download/Partitions.zip", stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        os.makedirs(sun_save_dir)

        sun_zip_path = os.path.join(sun_save_dir, "Partitions.zip")
        with open(sun_zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192): # Download in chunks for efficiency
                if chunk:
                    file.write(chunk)

        # Unzip file
        with zipfile.ZipFile(sun_zip_path, 'r') as zip_ref:
            zip_ref.extractall(sun_save_dir)

        # Delete zip
        delete_file(sun_zip_path)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading SUN397: {e}")
    except Exception as e:
        print(f"An unexpected error occurred downloading SUN397: {e}")
else:
    pass

print("Downloading SVHN..")
svhn_dir = os.path.join(data_root, "svhn")
if not os.path.exists(svhn_dir):
    os.makedirs(svhn_dir)
svhn_train = SVHN(root=svhn_dir, split="train", download=True)
svhn_test = SVHN(root=svhn_dir, split="test", download=True)

# Provide instructions for manual downloading of RESISC45, Stanford Cars, and ImageNet

