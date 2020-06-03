#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import PIL.Image as pil
import glob
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, help="video dataset directory")
parser.add_argument("--dst_dir", type=str, help="destination directory for the dataset")
parser.add_argument("--split_dir", type=str, help="directory containing the training/test split")
parser.add_argument("--camera_idx", action="store_true", help="if videos have the camera index appended")
parser.add_argument("--dataset", type=str, help="dataset name")
args = parser.parse_args()


# Construct raw dataset frame by frame
def read_video(file: str, src_folder: str, dst_folder: str, verbose: bool = False):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    src_path = os.path.join(src_folder, file)
    dst_path = os.path.join(dst_folder, file[:-6] if args.camera_idx else file[:-4] )
    cap = cv2.VideoCapture(src_path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # make destination folder
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        
    # frame index    
    frame_idx = 0    
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            frame = frame[:320, ...]
            
            # Display the resulting frame
            if verbose:
                cv2.imshow('Frame',frame)
            
            # save frame image
            img_path = os.path.join(dst_path, str(frame_idx) + ".png")
            frame_idx += 1
            img = pil.fromarray(frame[..., ::-1])
            img.save(img_path, 'png')

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

videos = os.listdir(args.src_dir)
videos = [v for v in videos if v.endswith('.mov')][:2]
for v in tqdm(videos):
    read_video(v, args.src_dir, args.dst_dir, False)

# get the path for all frames
frames_path = glob.iglob(os.path.join(args.dst_dir, '**','*.png'), recursive=True)
frames = []

for frame_path in tqdm(frames_path):
    # get a line
    line = frame_path.split("/")
    
    # split line in folder, scene and frame idx
    folder = "/".join(line[:-2])
    scene = line[-2]
    frame_idx = int(line[-1].split(".")[0])
    
    # we need the previous frame
    if frame_idx == 0:
        continue

    # get the next frame path
    fpath = os.path.join(folder, scene, str(frame_idx) + ".png")    
    frames.append(fpath)

formated_frames = []
for frame_path in frames:
    # get a line
    line = frame_path.split("/")
    
    # split line in folder, scene and frame idx
    folder = "/".join(line[:-2])
    scene = line[-2]
    frame_idx = line[-1].split(".")[0]
    new_line = " ".join([scene, frame_idx])
    formated_frames.append(new_line)

# Split dataset
with open(os.path.join(args.split_dir, "train_scenes.txt"), "rt") as fin:
    train_scenes = fin.read()
    train_scenes = train_scenes.split("\n")
    train_scenes = set(train_scenes)

with open(os.path.join(args.split_dir, "test_scenes.txt"), "rt") as fin:
    test_scenes = fin.read()
    test_scenes = test_scenes.split("\n")
    test_scenes = set(test_scenes)

train_frames = []
test_frames = []

for line in formated_frames:
    scene, frame = line.split(" ")

    if scene in train_scenes:
        train_frames.append(line)
    else:
        test_frames.append(line)


train_frames = "\n".join(train_frames)
test_frames = "\n".join(test_frames)

if not os.path.exists("splits"):
    os.mkdir("splits")

dst_path = os.path.join("splits", args.dataset)
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

with open(os.path.join(dst_path, "train_files.txt"), "wt") as fout:
    fout.write(train_frames)

with open(os.path.join(dst_path, "test_files.txt"), "wt") as fout:
    fout.write(test_frames)

