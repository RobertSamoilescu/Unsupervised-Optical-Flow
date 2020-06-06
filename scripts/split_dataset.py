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
parser.add_argument('--src_dir', type=str, help='directory containing all the video scenes')
parser.add_argument('--dst_dir', type=str, help='direcotry to save the split')
parser.add_argument('--camera_idx', action='store_true', help='if the camera index is present')
args = parser.parse_args()


if __name__ == "__main__":
    videos = os.listdir(args.src_dir)
    videos = [v for v in videos if v.endswith('.mov')]

    if args.camera_idx:
        scenes = [video[:-6] for video in videos]
    else:
        scenes = [video[:-4] for video in videos]

    test_scenes = np.random.choice(scenes, size=int(0.2 * len(scenes)))
    train_scenes = set(scenes).difference(test_scenes)

    train_scenes = "\n".join(train_scenes)
    test_scenes = "\n".join(test_scenes)

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
    
    with open(os.path.join(args.dst_dir, "train_scenes.txt"), "wt") as fout:
        fout.write(train_scenes)
    
    with open(os.path.join(args.dst_dir, "test_scenes.txt"), "wt") as fout:
        fout.write(test_scenes)