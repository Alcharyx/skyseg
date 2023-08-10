import torch
import torchvision
import onnxruntime
from utils.dataset import SkyFinderDataset
from torch.utils.data import DataLoader
from utils.Unet import UNET

def save_checkpoint(state, filename="my_checkpoint.pt"):
    """
    Save the current model state into a file

    """
    print("=> Saving checkpoint")
    torch.save(state,filename )

def export_to_ONNX(state,batch_size, channels, height, width,filename):
    """
    export a saved checkpoint to the ONNX format

    """
    dummy_input = torch.randn(batch_size, channels, height, width)
    torch.onnx.export(state,dummy_input, filename, verbose= False)

def load_ONNX(model_path):
    """
    load a ONNX model file

    """
    return onnxruntime.InferenceSession(model_path)

def load_checkpoint(checkpoint, model):
    """
    Load a checkpoint file
    
    """
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 4,
        pin_memory = True):
    """
    Generate training and validation data loader

    Parameters
    ----------

    train_dir : string
        Path of the directory of training images with the name starting by the id of the image ex :"615_XXXX.jpg" to reference to the correct mask
    train_mask_dir : string
        Path of the directory of training masks named with their id in the .png format ex: "615.png", mask must be the same size as the image
    val_dir : string
        Path of the directory of validation images with the name starting by the id of the image ex :"615_XXXX.jpg" to reference to the correct mask
    val_mask_dir : string
        Path of the directory of validation masks named with their id in the .png format ex: "615.png", mask must be the same size as the image
    batch_size : int
        size of a batch during  training
    train_transform : albumentation.compose
        A composition of the different data augmentation and processing for training
    val_transform : albumentation.compose
        A composition of the different data processing for validation (resize and normalize only) 
    num_workers : int, optional
        Number of worker to load the data
    pin_memory : bool, optional
        Enable pin memory for more efficient data transfer host <=> device
    
    """
    train_ds = SkyFinderDataset(image_dir= train_dir,
                                mask_dir = train_maskdir,
                                transform= train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers= num_workers,
        pin_memory= pin_memory,
        shuffle= True,
    )

    val_ds = SkyFinderDataset(
        image_dir= val_dir,
        mask_dir= val_maskdir,
        transform= val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory= pin_memory,
        shuffle= False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    """
    Get accuracy and DSC value
    
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x,y in loader :
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / (
                     (preds + y).sum() + 1e-8
                )
            
    print(
         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}"
    )
    print(
         f"Dice Score: {dice_score/len(loader):.2f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder = "saved_images/",device = "cuda"):
        """
        Save predictions images and ground truth
        
        """
        model.eval()
        for idx, (x,y) in enumerate(loader):
            x = x.to(device = device)
            with torch.no_grad():
                 preds = torch.sigmoid(model(x))
                 preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/pred_{idx}_original.png")

