
import os
import re
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.RIFE import Model
import argparse
from model.pretrained_RIFE_loader import IFNet_update, convert_load
import time



def load_frames(frame_folder, start_frame, num_frames=4):
    frames = []
    for i in range(1, num_frames+1):
        frame_path = os.path.join(frame_folder, f"frame_{start_frame + i:04d}.jpg")
        frame = Image.open(frame_path).convert('RGB')
        frames.append(frame)
        if i != num_frames:
            frames.append(frame)
    return frames

def preprocess_frames(frames):
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = torch.stack([transform(frame) for frame in frames], dim=0)
    tensor = tensor.view(1, 3*7, tensor.shape[2], tensor.shape[3])
    del frames  # Free memory
    torch.cuda.empty_cache()  # Clear CUDA cache
    return tensor.to(device)

def count_frames(frame_folder, pattern=r"frame_\d{4}\.jpg"):
    # List all files in the directory
    all_files = os.listdir(frame_folder)
    
    # Filter files that match the frame naming pattern
    frame_files = [f for f in all_files if re.match(pattern, f)]
    
    # Sort the frame files to ensure they are in sequence
    frame_files.sort()
    
    # Return the count of frames
    return len(frame_files)

def save_frame(tensor, output_folder, frame_index):
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu().detach())
    img.save(os.path.join(output_folder, f"frame_{frame_index:04d}.jpg"))
    del img  # Free memory
    torch.cuda.empty_cache()  # Clear CUDA cache

def print_memory_usage(phase):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    allocated_GB = allocated / (1024)
    reserved_GB = reserved / (1024)
    print(f"{phase} \n- Allocated Memory: {allocated_GB} GB({allocated:.2f} MB), \n- Reserved Memory: {reserved_GB} GB({reserved:.2f} MB)")

def inference_video(model, frame_folder, output_folder, total_frames):

    for start_frame in range(0,total_frames - 2 + 1, 3):  # Adjust the step to handle overlap or gaps
        i = int(start_frame/3)
        if start_frame + 4 > total_frames:
            print("Out of frames")
            break
        frames = load_frames(frame_folder, start_frame)
        save_start_point = i*6
        input_tensor = preprocess_frames(frames)
        #print('gointo model')
        output_allframes_tensors = model(input_tensor)
        #print('compute finished')
        interpolated_frames = output_allframes_tensors[:-1] 
        for i in range(6):
            save_frame(interpolated_frames[i, :, :, :], output_folder, save_start_point + i + 1)
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    args = parser.parse_args()
    torch.cuda.set_device(0)  # Set the appropriate CUDA device
    torch.backends.cudnn.benchmark = True
    
    # Config
    root = '/root/MyCode/Valid/image_pyramid/Upload_image_pyramid/'
    output_root = "/root/autodl-tmp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # frame_path = "/root/autodl-fs/Origin/soccer"
    frame_path = output_root + "/outputs_image_pyramid_soccer/soccer_4x"

    output_path = output_root + "/outputs_image_pyramid_soccer/soccer_8x"
    pretrained_model_path = '/root/MyCode/Valid/image_pyramid/Upload_image_pyramid/intrain_log'
    
    
    
    
    
    start_time = time.time()
    pretrained_path = 'RIFE_log'
    checkpoint = convert_load(torch.load(f'{pretrained_path}/flownet.pkl', map_location=device))
    Ori_IFNet_loaded = IFNet_update()
    Ori_IFNet_loaded.load_state_dict(checkpoint)
    for param in Ori_IFNet_loaded.parameters():
        param.requires_grad = False
    
    model = Model(Ori_IFNet_loaded, 0)
    model.load_model(pretrained_model_path)

    model.eval()
    print_memory_usage("Loaded, Start Compute")
    num_frames = count_frames(frame_path)
    inference_video(model.simple_inference, frame_path, output_path, num_frames)
    
    end_time = time.time()
    print("Doubled ", num_frames, " frames")
    print("Time spent for the interpolation:", round((end_time - start_time) / 60, 2), "minutes")   
    