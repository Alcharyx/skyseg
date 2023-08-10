import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.Unet import UNET

from utils.model_utils import (
     load_checkpoint,
     save_checkpoint,
     export_to_ONNX,
     get_loaders,
     check_accuracy,
     save_predictions_as_imgs
 )

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "train_images/"
TRAIN_MASK_DIR = "train_mask/"
VAL_IMG_DIR = "val_images/"
VAL_MASK_DIR = "val_mask/"


def train(loader, model, optimizer, loss_fn, scaler):
    """
    Run the training of the model

    Parameters
    ----------
    loader :  torch.utils.data.Dataloader
        training data loader
    model : torch model
        Instance of the model to be trained
    optimizer : torch optimizer
        Optimizer used for the training
    loss_fn : torch loss
        Loss function used for the training
    scaler : torch scaler
        Scaler gradient used for the training

    """
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())


def main():
    """
    Run the Unet training with basic parameters, evaluate the model and save a checkpoint every epochs, save prediction image compared with the original mask

    """

    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 20, p = 1.0),
            A.HorizontalFlip(p= 0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels = 3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pt"), model)
        #check_accuracy(val_loader,model,device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)


        # save model
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,filename="my_checkpoint.pt")

        #check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
    
    #save test example
    save_predictions_as_imgs(
        val_loader,model,folder = "saved_images/", device = DEVICE
    )

if __name__ == "__main__":
    
    main()
    #export_to_ONNX("my_checkpoint.pt", 1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, "my_UNET.onnx", DEVICE)