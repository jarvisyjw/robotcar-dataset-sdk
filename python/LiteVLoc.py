import numpy as np
from matplotlib import pyplot as plt
from transform import so3_to_quaternion, se3_to_components, build_se3_transform, build_se3_transform_from_xyzquaternion
from evo.tools import file_interface
from scipy.spatial.transform import Rotation 
from pathlib import Path
import os


def plot_utms(eastings, northings, title="UTM Coordinates Plot"):
    """
    Plots UTM coordinates on a 2D graph.

    Args:
        eastings (list or array): List of easting (x) values in meters.
        northings (list or array): List of northing (y) values in meters.
        title (str): Title of the plot (default: "UTM Coordinates Plot").
    """
    if len(eastings) != len(northings):
        raise ValueError("Eastings and Northings lists must have the same length.")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(eastings, northings, c='blue', label='UTM Points', alpha=0.7)
    plt.plot(eastings, northings, linestyle='--', color='gray', alpha=0.5, label='Path')
    plt.xlabel("Easting (meters)")
    plt.ylabel("Northing (meters)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensures equal scaling for both axes
    plt.show()


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

# plot the poses
def plot_poses(poses, current_idx=None):
    poses_tum = [se3_to_components(pose, quaternion=True) for pose in poses]
    poses_tum = np.array(poses_tum)
    xyz = poses_tum[:, :3]
    plt.figure(figsize=(10, 10))
    plt.title("Interpolated Poses")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    if current_idx:
        plt.scatter(xyz[current_idx, 0], xyz[current_idx, 1], color='red', label='Current Pose')
        plt.legend()
    plt.grid()
    plt.plot(xyz[:, 0], xyz[:, 1], 'o-')
    plt.show()
    
# convert poses to TUM format
def poses_to_tum(poses, timestamps):
    """
    Converts poses to TUM format.

    Parameters:
    - poses: List of poses in SE3 format.
    - timestamps: List of timestamps corresponding to the poses.

    Returns:
    - tum_format: List of strings in TUM format.
    """
    tum_format = []
    for i, pose in enumerate(poses):
        components = se3_to_components(pose, quaternion=True)
        timestamp = timestamps[i]
        tum_format.append(f"{timestamp} {components[0]} {components[1]} {components[2]} {components[3]} {components[4]} {components[5]} {components[6]}")
    return tum_format

# convert poses to map-free format
def poses_to_map_free(poses, timestamps):
    """
    Converts poses to map-free format.

    Parameters:
    - poses: List of poses in SE3 format.
    - timestamps: List of timestamps corresponding to the poses.

    Returns:
    - map_free_format: List of strings in map-free format.
    """
    map_free_format = []
    for i, pose in enumerate(poses):
        components = se3_to_components(pose, quaternion=True)
        timestamp = timestamps[i]
        map_free_format.append(f"{timestamp} {components[0]} {components[1]} {components[2]} {components[3]} {components[4]} {components[5]}")
    return map_free_format

def write_to_litevloc_format(frames_file, poses_file, output_file):
    """
    Writes poses and frames to a LiteVLoc format file.

    Parameters:
    - frames_file: Path to the frames file.
    - poses_file: Path to the poses file.
    - output_file: Path to save the output file in LiteVLoc format.
    """
    timestamps = []
    img_paths = []
    
    with open(frames_file, 'r') as f:
        frames = f.readlines()
    for frame in frames:
        tokens = frame.strip().split()
        if len(tokens) < 2:
            continue
        timestamps.append(int(tokens[0]))
        img_paths.append(tokens[1].split('seq/', 1)[-1])  # Get the image file name
    
    with open(poses_file, 'r') as f:
        poses = f.readlines()
    if len(timestamps) != len(poses):
        raise ValueError("Number of timestamps does not match number of poses.")
    if len(timestamps) != len(frames):
        raise ValueError("Number of timestamps does not match number of frames.")
    if len(img_paths) != len(frames):
        raise ValueError("Number of image paths does not match number of frames.")
    
    with open(output_file, 'w') as f:
        for img_path, pose in zip(img_paths, poses):
            ts, tx, ty, tz, qx, qy, qz, qw = pose.strip().split()
            qx, qy, qz = 0, 0, 0  # Set qx, qy, qz to zero as per the original code
            qw = 1.0
            # se3_poses = build_se3_transform_from_xyzquaternion(np.array([float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)]))
            se3_poses = convert_vec_to_matrix(np.array([float(tx), float(ty), float(tz)]), 
                                                np.array([float(qx), float(qy), float(qz), float(qw)]), mode='xyzw')
            # inverse the pose to get the camera frame
            # se3_poses_inv = se3_poses.copy()
            se3_poses_inv = np.linalg.inv(se3_poses)
            trans, quat = convert_matrix_to_vec(se3_poses_inv, mode='xyzw')
            # qw, qx, qy, qz = so3_to_quaternion(se3_poses_inv[:3, :3])
            # tx, ty, tz = float(se3_poses_inv[0, 3]), float(se3_poses_inv[1, 3]), float(se3_poses_inv[2, 3])            # print(f"{tx}, {ty}, {tz}")
            f.write(f"seq/{img_path} {quat[3]} {quat[0]} {quat[1]} {quat[2]} {trans[0]} {trans[1]} {trans[2]}\n")
            
def write_intrinsics_to_litevloc_format(frames_file, output_file):
    """
    Writes camera intrinsics to a LiteVLoc format file.

    Parameters:
    - frames_file: Path to the frames file.
    - output_file: Path to save the output file in LiteVLoc format.
    """
    timestamps = []
    img_paths = []
    
    with open(frames_file, 'r') as f:
        frames = f.readlines()
    for frame in frames:
        tokens = frame.strip().split()
        if len(tokens) < 2:
            continue
        timestamps.append(int(tokens[0]))
        img_paths.append(tokens[1].split('seq/', 1)[-1])  # Get the image file name
    
    fx, fy, cx, cy, image_width, image_height = 605.128601, 604.97406, 520.45343, 393.878479, 1024, 768
    with open(output_file, 'w') as f:
        for img_path in img_paths:
            f.write(f"seq/{img_path} {fx} {fy} {cx} {cy} {image_width} {image_height}\n")
    

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
    n = min(traj1.num_poses, traj2.num_poses)
    n = 600
    # Align the trajectories using a rigid transformation
    traj1.align_origin(traj2)
    r_a, t_a, s = traj1.align(traj2, correct_scale=correct_scale, n = n)
    # traj1_align = traj1.align_origin(traj2)
    file_interface.write_tum_trajectory_file(output_file_path, traj1)
    # file_interface.write_tum_trajectory_file(output_file_path, traj1)
    # Print results
    print(f"Aligned trajectory saved to: {output_file_path}")

def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Align two trajectories using a rigid transformation.")
    parser.add_argument("--file1", type=str, help="Path to the first trajectory file.")
    parser.add_argument("--file2", type=str, help="Path to the second trajectory file.")
    parser.add_argument("--output_file", type=str, help="Path to save the aligned trajectory.")
    parser.add_argument("--correct_scale", action='store_true', help="Correct scale during alignment.")
    return parser

def convert_pose_to_camera_frame(input_file, output_file, transform):
    """
    Convert poses from a file to the camera frame using a given transformation.

    Parameters:
    - input_file: Path to the input file containing poses.
    - output_file: Path to save the converted poses.
    - transform: SE3 transformation matrix to apply to the poses.
    """
    trajectory = file_interface.read_tum_trajectory_file(input_file)
    trajectory.transform(transform, right_mul=True)
    print(trajectory)
    
    file_interface.write_tum_trajectory_file(output_file, trajectory)
    

def read_poses(dataset_folder):
    # pose_path = dataset_folder
    pose_path = Path(dataset_folder, 'poses_abs_gt.txt')
    # pose_path = Path(dataset_folder, 'trajectory_align.tum')
 
    # if not Path(pose_path).exists():
    #     pose_path = Path(dataset_folder, 'trajectory_align.tum')
    #     print(f"File not found: {pose_path}")
    #     return None
 
    data_dict = {}
    with open(pose_path, 'r') as f:
        for line_id, line in enumerate(f):
            if line.startswith('#'):
                continue
 
            parts = line.strip().split()
            if 'jpg' in parts[0] or 'png' in parts[0]:
                # provide img_name
                img_name = parts[0]
                data = list(map(float, parts[1:]))
            else:
                # not provide img_name
                img_name = "seq/{frame_id:06d}.color.jpg".format(frame_id=line_id)
                data = list(map(float, parts))
 
            data_dict[img_name] = np.array(data)
            
    return data_dict


def convert_matrix_to_vec(
    transform: np.ndarray,
    mode: str = 'xyzw'):
    """Convert 4x4 transformation matrix to translation and quaternion.
    
    Args:
        transform: 4x4 transformation matrix
        mode: Desired quaternion format ('xyzw' or 'wxyz')
        
    Returns:
        Tuple of (translation, quaternion)
    """
    if transform.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix")
        
    translation = transform[:3, 3]
    rotation = Rotation.from_matrix(transform[:3, :3])
    quat = rotation.as_quat()
    
    if mode == 'wxyz':
        quat = np.roll(quat, 1)
        
    return translation, quat

def convert_vec_to_matrix(
    translation: np.ndarray,
    quaternion: np.ndarray,
    mode: str = 'xyzw'
) -> np.ndarray:
    """Convert translation and quaternion to 4x4 transformation matrix.
    
    Args:
        translation: [x, y, z] translation vector
        quaternion: Quaternion components
        mode: Quaternion format ('xyzw' or 'wxyz')
        
    Returns:
        4x4 transformation matrix
    """
    tf = np.eye(4)
    tf[:3, 3] = translation
    
    if mode not in ['xyzw', 'wxyz']:
        raise ValueError(f"Invalid quaternion mode: {mode}")
        
    if mode == 'wxyz':
        quaternion = np.roll(quaternion, -1)
        
    tf[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return tf


def deleate_images_not_in_list(folder_path, file_path):
    """
    Deletes images in a folder that are not listed in a specified text file.
    
    The text file should contain the names of images to keep, one per line.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    images_to_keep = []
    for line in lines:
        line = line.strip()
        ts, img_name = line.split(' ')
        img_name = img_name.split('seq/', 1)[-1]  # Get the image file name
        images_to_keep.append(img_name)

    # Iterate through the folder and remove files not in the list
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is not in the keep list and is a file
        if filename not in images_to_keep and os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {filename}")

    print("Cleanup complete!")

if __name__ == "__main__":
    args = parser().parse_args()
    # deleate_images_not_in_list(args.file1, args.file2)
    
    write_intrinsics_to_litevloc_format(args.file1, args.output_file)
    # T_camera_vehicle = build_se3_transform(np.array([0, 0, 0, 0, 0, 0]))
    # T_camera_vehicle = np.eye(4)
    # T_camera_vehicle[:3, :3] = np.array([[0, 1, 0],
    #                                      [0, 0, 1],
    #                                      [1, 0, 0]])  # Rotation to align camera frame with vehicle frame
    # T_vehicle_ins = build_se3_transform(np.array([-1.7132, 0.1181, 1.1948, -0.0125, 0.0400, 0.0050]))  # Identity transformation
    # T_camera_ins = T_camera_vehicle * T_vehicle_ins
    # print(f"T_camera_ins: {T_camera_ins}")
    # convert_pose_to_camera_frame(args.file1, args.output_file, T_camera_ins)
    
    # T = np.eye(4)
    # T[:3, :3] = np.array([[0, 0, 1],
    #                       [1, 0, 0],
    #                       [0, 1, 0]])
    
    # convert_pose_to_camera_frame(args.file1, args.output_file, T)
    
    # write_to_litevloc_format(args.file1, args.file2, args.output_file)
    # align_trajectory(args.file1, args.file2, args.output_file, args.correct_scale)
    
    # database_folder = Path(args.file1)
    # query_folder = Path(args.file2)
    # db_poses = read_poses(database_folder)
    # query_poses = read_poses(query_folder)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # i = 0
    # for query_img_name, query_pose in query_poses.items():
    #     # ax.plot(query_pose[1], query_pose[2], 'ro', markersize=5, label='Query Pose' if query_img_name == list(query_poses.keys())[0] else "")
    #     Tc2w = convert_vec_to_matrix(query_pose[4:], query_pose[:4], 'wxyz')
    #     trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    #     ax.plot(trans_query[0], trans_query[1], 'ro', markersize=5, label='Query Pose' if query_img_name == list(query_poses.keys())[0] else "")
    #     # print(query_pose)
    #     # print(trans_query, quat_query)
    #     # if i > 5:
    #     #     break
    #     # i +=1
    #     # exit(0)
    #     # if query_img_name not in db_poses:
    #     #     print(f"Image {query_img_name} not found in database poses.")
    #     #     continue
    #     # db_pose = db_poses[query_img_name]
    #     # Tc2w = convert_vec_to_matrix(query_pose[4:], query_pose[:4], 'wxyz')
    # for db_img_name, db_pose in db_poses.items():
    #     # ax.plot(db_pose[1], db_pose[2], 'bo', markersize=5, label='Reference Pose' if db_img_name == list(db_poses.keys())[0] else "")
    #     Tc2w = convert_vec_to_matrix(db_pose[4:], db_pose[:4], 'wxyz')
    #     trans_ref, quat_ref = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    #     # # print(trans_ref, quat_ref)
    #     # print(db_pose)
    #     # if i > 5:
    #     #     break
    #     # i +=1
        
    #     ax.plot(trans_ref[0], trans_ref[1], 'bo', markersize=5, label='Reference Pose' if db_img_name == list(db_poses.keys())[0] else "")  
    
    # ax.set_title("Query and Reference Poses")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.grid()
    # ax.axis('equal')  # Ensures equal scaling for both axes
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    
    # db_pose = test_ds.database_poses[db_img_name]
    # Tc2w = convert_vec_to_matrix(query_pose[4:], query_pose[:4], 'wxyz')
    # trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    # Tc2w = convert_vec_to_matrix(db_pose[4:], db_pose[:4], 'wxyz')
    # trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    # trans_err, rot_err = compute_pose_error((trans_query, quat_query), (trans_db, quat_db), mode='vector')
    
    
    
    
    
    
 
 
 


