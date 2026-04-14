################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#  Jingwen Yu (jyubt@connect.ust.hk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
from tqdm import tqdm

from transform import se3_to_components
from interpolate_poses import interpolate_ins_poses
from image import load_image
from camera_model import CameraModel
from LiteVLoc import sample_poses_by_distance, plot_poses, poses_to_tum

def main():
    parser = argparse.ArgumentParser(description='Extract images from a given directory')
    parser.add_argument('dir', type=str, help='Directory containing images.')
    parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
    parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')
    parser.add_argument('--save_dir', type=str, default=None, help='(optional) Path to save the extracted images. If not provided, images will not be saved.')
    args = parser.parse_args()
    
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    else:
        print("No save path provided, images will not be saved.")
    
    camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)
    timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
    ins_path = os.path.join(os.path.join(args.dir, os.pardir, os.pardir, 'rtk.csv'))
    use_rtk = True
    # print(f"{ins_path}")
    
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
    if not os.path.isfile(timestamps_path):
        raise IOError("Could not find timestamps file")

    if not os.path.isfile(ins_path):
        ins_path = os.path.join(args.dir, os.pardir, os.pardir, 'gps/ins.csv')
        use_rtk = False
    if not os.path.isfile(ins_path):
        raise IOError(f"Could not find ins file: {ins_path}")
    
    model = None
    if args.models_dir:
        model = CameraModel(args.models_dir, args.dir)
    else:
        print("No camera model provided, images will not be undistorted.")

    # load timestamps
    timestamps = load_timestamps(timestamps_path)
    if not timestamps:
        print("No timestamps found in the file.")
        return
    
    # interpolate poses
    poses, timestamps = interpolate_ins_poses(ins_path, timestamps, timestamps[0], use_rtk=use_rtk)
    print(f"Interpolated poses: {len(poses)} poses for {len(timestamps)} timestamps")
    
    # sample poses by distance
    idx, sampled_poses = sample_poses_by_distance(poses, min_distance = 3.9)
    sample_timestamps = [timestamps[i] for i in idx]
    print(f"Sampled poses: {len(sampled_poses)} poses for {len(idx)} timestamps with a minimum distance of 3.9 meters")
    
    # plot poses for debugging
    plot_poses(sampled_poses)
    
    # extract and display images
    run(args.dir, sample_timestamps, model, sampled_poses, args.save_dir)

def load_timestamps(timestamps_path):
    timestamps = []
    with open(timestamps_path) as f:
        for line in f:
            tokens = line.split()
            timestamp = int(tokens[0])
            # chunk = int(tokens[1])
            timestamps.append(timestamp)
    return timestamps

def run(dir, timestamps, model, poses, save_dir=None):
    xyzqxqyqzqw = np.array([se3_to_components(pose) for pose in poses])
    fig, ax = plt.subplots(1,2)
    ax[1].plot(xyzqxqyqzqw[:, 0], xyzqxqyqzqw[:, 1], color='b')
    ax[1].set_title("Position")
    ax[1].set_xlabel("X (m)")
    ax[1].set_xlabel("X (m)")
    ax[1].grid()
    ax[0].set_title("Current View")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # save trajectory
    # tum_format
    poses_tum = poses_to_tum(poses, timestamps)
    
    if save_dir:
        tum_path = os.path.join(save_dir, 'trajectory.tum')
        with open(tum_path, 'w') as tum_file:
            for line in poses_tum:
                tum_file.write(line + '\n')
        img_path = os.path.join(save_dir, 'seq')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        frames_file = open(os.path.join(save_dir, 'frames.txt'), 'w')
    
    for t in tqdm(timestamps):
        filename = os.path.join(dir, str(t) + '.png')
        img = load_image(filename, model)
        if save_dir:
            save_path = os.path.join(img_path, str(t) + '.color.jpg')
            plt.imsave(save_path, img)
            frames_file.write(f"{t} {save_path}\n")
        ax[0].imshow(img)
        ax[1].scatter(xyzqxqyqzqw[timestamps.index(t),0], xyzqxqyqzqw[timestamps.index(t),1], color='red', label='Current Pose')
        # plt.pause(0.001)

if __name__ == "__main__":
    main()