import cv2
from thread_videocapture import VideoCapture
from tqdm import tqdm
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy as copy
from utils.Unet import UNET
from utils.model_utils import (
    load_checkpoint,    
)

def main(video_path,
         model_path,
         resolution,
         device = "cuda"):
    """
    Run the inference on a video, resized to desired dimensions and plot the segmentation mask on top of the original image

    Parameters
    ----------
    video_path : string
        Path of the video to run the inference on
    model_path : string
        Path of the model weights
    resolution : list[int,int]
        Resolution to resize the image before the inference
    device : string, optional
        Name of the device used for the inference
    """
        
    model = UNET(in_channels = 3, out_channels=1).to(device)
    load_checkpoint(torch.load(model_path), model)
    cap_resize = A.Compose(
        [
            A.Resize(height = resolution[1], width = resolution[0]),
        ],
    )
    cap_preprocess = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    cap = VideoCapture(video_path,
                       target_fps=10,
                       tqdm_progressbar= tqdm())
    try:
        while(cap_frame := cap.read()[1]) is not None:
            
            resize_frame = cap_resize(image = cap_frame)["image"]
            overlay = np.zeros_like(resize_frame)
            frame = resize_frame.astype(np.float32)
            frame = cap_preprocess(image = frame)["image"].unsqueeze(0)
            frame = frame.to(device = device)

            with torch.no_grad():
                predictions = torch.sigmoid(model(frame))
                predictions = (predictions > 0.75).float()
            
            predictions = predictions.squeeze().cpu().numpy()  # Remove singleton dimensions
            predictions = (predictions * 255).astype(np.uint8)    
            overlay[:,:,0] = predictions

            opacity = 0.4
            img_final = cv2.addWeighted(resize_frame, opacity , overlay, 1- opacity,0)
            cv2.imshow("Inference Result", img_final)
            cv2.imshow("Original Image",resize_frame)
            cv2.waitKey(1)
    finally:
        cap.release()

if __name__ == "__main__":
    main(video_path="landscape_test_vid.mp4",
         model_path="my_checkpoint.pt",
         resolution=[640,480],
         device="cuda")