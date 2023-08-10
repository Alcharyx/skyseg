import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SkyFinderDataset(Dataset):
    """
    SKyFinderDataset loader, download the dataset : https://zenodo.org/record/5884485

    Parameters
    ----------
    image_dir : string,
        Path of the directory of images with the name starting by the id of the image ex :"615_XXXX.jpg" to reference to the correct mask
    mask_dir : string
        Path of the directory of masks named with their id in the .png format ex: "615.png", mask must be the same size as the image
    transform : albumentation.compose
        A composition of the different data augmentation with their probabilities during the training and data processing  
    """
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.images[index].split("_")[0] + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype= np.float32)
        mask[mask == 255.0] = 1.0 #for sigmoid acitvation function
        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image,mask