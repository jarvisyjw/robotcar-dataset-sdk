"""Scripts used to generate data for LiteVLoc

"""
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from interpolate_poses import interpolate_ins_poses
from transform import so3_to_quaternion, se3_to_components
from evo.tools import file_interface
# from evo.core.trajectory import PoseTrajectory3D
# from evo.core.transformations import quaternion_matrix
import shutil

def align_trajectory(file1_path, file2_path, output_file_path, correct_scale=False):
    """
    Align two trajectories using a rigid transformation.

    Parameters:
    - traj1: First trajectory (numpy array of shape (N, 7) or (N, 6)).
    - traj2: Second trajectory (numpy array of shape (M, 7) or (M, 6)).
    - correct_scale: If True, scale the first trajectory to match the second.

    Returns:
    - aligned_traj1: Aligned first trajectory.
    - transform: Transformation matrix used for alignment.
    """
    # Load the trajectories
    traj1 = file_interface.read_tum_trajectory_file(file1_path)
    traj2 = file_interface.read_tum_trajectory_file(file2_path)

    # Align the trajectories using a rigid transformation
    r_a, t_a, s = traj1.align(traj2, correct_scale=correct_scale)
    file_interface.write_tum_trajectory_file(output_file_path, traj1)
    # Print results
    print(f"Aligned trajectory saved to: {output_file_path}")

def sample_poses_by_distance(poses, min_distance):
    """
    Samples poses by a minimum distance threshold.

    Parameters:
    - poses: 
    - min_distance: Minimum distance between consecutive sampled poses.

    Returns:
    - sampled_poses: List of sampled poses.
    """
    sampled_idx = []
    sampled_idx.append(0)  # Start with the first pose
    sampled_poses = []
    last_position = poses[0][:3, 3]  # Extract the translation part of the first pose
    sampled_poses.append(poses[0])  # Add the first pose to the sampled poses\
        
    for idx, pose in enumerate(poses[1:]):
        # Extract the translation part of the pose
        translation = pose[:3, 3]
        distance = np.linalg.norm(translation - last_position)
        if distance >= min_distance:
            sampled_poses.append(poses[idx+1])
            last_position = translation
            sampled_idx.append(idx+1)

    return sampled_idx, sampled_poses

def extract_timestamps_from_images(directory):
    timestamps = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .jpg image
        if filename.endswith(".jpg"):
            # Extract the numeric part (timestamp) before the .jpg
            timestamp = filename.split(".")[0]
            if timestamp.isdigit():
                timestamps.append(int(timestamp))
    
    # Sort the timestamps in ascending order
    timestamps.sort()
    return timestamps

## load timestamps from txt to list
def load_timestamps_from_txt(file_path):
    """
    Load timestamps from a text file into a list.
    
    Args:
        file_path (str): Path to the text file containing timestamps.
    
    Returns:
        list[int]: List of timestamps.
    """
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines and non-digit lines
            # if line.strip() and line.strip().isdigit():
            #     continue
            if line.startswith("#"):
                continue
            timestamps.append(int(line.strip()))
        # timestamps = [int(line.strip()) for line in file if line.strip().isdigit()]
    return sorted(timestamps)

def interpolate_timestamps(rtk_path, target_timestamps, origin_timestamp):
    """
    Interpolate timestamps to match the target timestamps.
    
    Args:
        timestamps (list[int]): List of timestamps extracted from image filenames.
        target_timestamps (list[int]): List of target timestamps for interpolation.
    
    Returns:
        list[int]: Interpolated timestamps.
    """
    return interpolate_ins_poses(rtk_path, target_timestamps, origin_timestamp, use_rtk=True)

def write_names(input_file, output_file):
    """
    Write names from input file to output file.
    
    Args:
        input_file (str): Path to the input file containing names.
        output_file (str): Path to the output file to save names.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():  # Skip empty lines
                data = line.strip().split()
                assert len(data) == 8, "Each line must contain 8 elements."
                timestamp = int(float(data[0]))
                pose = data[1:]
                name = f"seq/{timestamp}.jpg"
                outfile.write(f"{name} {str(pose[0])} {str(pose[1])} {str(pose[2])} {str(pose[3])} {str(pose[4])} {str(pose[5])} {str(pose[6])}\n")

def load_camera_poses(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            img_path = str(parts[0])
            translation = list(map(float, parts[1:4]))
            quaternion = list(map(float, parts[4:8]))
            data.append((img_path, translation, quaternion))
    return data

def copy_imgs(root_path, image_list_file_path, output_dir):
    """
    Copies images from the input list to the output directory.

    Parameters:
    - image_list: List of image file paths.
    - output_dir: Directory where the images will be copied.

    Returns:
    - None
    """

    img_list = load_camera_poses(image_list_file_path)
    # name_file = os.path.join(output_dir, 'poses_abs_gt.txt')
    
    # fout = open(name_file, 'w')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    for data in img_list:
        img_path, trans, quat = data
        img_path_target = img_path.split('/')[-1]
        shutil.copy(os.path.join(root_path, img_path_target), os.path.join(output_dir, img_path))
        
    # fout.close()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and sort timestamps from image filenames.")
    parser.add_argument("--directory", type=str, help="Directory containing the image files.")
    parser.add_argument("--output", type=str, default="timestamps.txt", help="Output file to save sorted timestamps.")
    parser.add_argument("--rtk", type=str, help="Path to RTK file for interpolation.")
    parser.add_argument("--timestamps", type=str, help="Path to file containing timestamps for interpolation.")
    parser.add_argument("--align", nargs=2, type=str, help="Paths to two trajectory files to align.")
    parser.add_argument("--output_file", type=str, default="output", help="Output directory for results.")
    parser.add_argument("--input_file", type=str, default="output", help="Output directory for results.")
    parser.add_argument("--root_dir", type=str)

    args = parser.parse_args()
    # directory_path = args.directory  # Replace with your directory path
    # sorted_timestamps = extract_timestamps_from_images(directory_path)
    # print("Sorted Timestamps:", sorted_timestamps)
    # np.savetxt(os.path.join(args.output, "timestamps.txt"), sorted_timestamps, fmt='%d', header='Timestamps extracted from images', comments='')
    if args.directory:
        sorted_timestamps = extract_timestamps_from_images(args.directory)
        np.savetxt(args.output, sorted_timestamps, fmt='%d', header='Timestamps extracted from images', comments='')
        print(f"Sorted timestamps saved to {args.output}")
    if args.rtk:
        target_timestamps = load_timestamps_from_txt(args.timestamps) if args.timestamps else sorted_timestamps
        # print(f"Target timestamps for interpolation: {target_timestamps}")
        origin_timestamp = target_timestamps[0]
        poses, timestamps = interpolate_timestamps(args.rtk, target_timestamps, origin_timestamp)
        print(f"Interpolated poses: {len(poses)} poses for {len(timestamps)} timestamps")

        idx, sampled_poses = sample_poses_by_distance(poses, min_distance = 3.9)
        sample_timestamps = [timestamps[i] for i in idx]
        print(f"Sampled poses: {len(sampled_poses)} poses for {len(idx)} timestamps with a minimum distance of 3.9 meters")
        # print(f'{idx}')

        # plot the poses
        poses_tum = [se3_to_components(pose, quaternion=True) for pose in sampled_poses]
        poses_tum = np.array(poses_tum)
        xyz = poses_tum[:, :3]
        plt.figure(figsize=(10, 10))
        plt.title("Interpolated Poses")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid()
        plt.plot(xyz[:, 0], xyz[:, 1], 'o-')
        plt.savefig(os.path.join(args.output, "interpolated_poses.png"))
        
        output_file = os.path.join(args.output, "interpolated_poses.txt")
        with open(output_file, 'w') as f:
                    for pose, ts in zip(poses_tum, sample_timestamps):
                        f.write(f"{ts} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n")
        print(f"Interpolated poses saved to {output_file}")
    if args.align:
        traj1_path = args.align[0]
        traj2_path = args.align[1]
        output_file_path = args.output_file if len(args.align) < 3 else args.align[2]
        align_trajectory(traj1_path, traj2_path, output_file_path)
        print(f"Aligned trajectory saved to {output_file_path}")
    # if args.input_file:
    #     input_file = args.input_file
    #     output_file = os.path.join(args.output_file)
    #     write_names(input_file, output_file)
    #     print(f"Names written to {output_file}")
    
    if args.root_dir:
        root_path = args.root_dir
        image_list_file_path = os.path.join(args.input_file)
        output_dir = args.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        copy_imgs(root_path, image_list_file_path, output_dir)
        print(f"Images copied to {output_dir}")