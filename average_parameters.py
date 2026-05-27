import numpy as np
from scipy.spatial.transform import Rotation as R

extrinsics = np.load("calibration_images/camera_params/extrinsics.npy")
intrinsics = np.load("calibration_images/camera_params/intrinsics.npy")

num_frames = extrinsics.shape[0]
num_cameras = extrinsics.shape[1]

print(f" Number of frames: {num_frames} \n Number of cameras: {num_cameras}")

avg_extrinsics = np.zeros((num_cameras, 3, 4))
avg_intrinsics = np.zeros((num_cameras, 3, 3))

for cam_idx in range(num_cameras):
    print(f"Processing camera {cam_idx + 1}/{num_cameras}")

    rotations = []
    translations = []

    for frame_idx in range(num_frames):
        ext = extrinsics[frame_idx, cam_idx]

        R_mat = ext[:, :3]
        t_vec = ext[:, 3]

        rotations.append(R_mat)
        translations.append(t_vec)

    rot_obj = R.from_matrix(rotations)
    quats = rot_obj.as_quat()

    ref_quat = quats[0]
    for i in range(len(quats)):
        if np.dot(ref_quat, quats[i]) < 0:
            quats[i] *= -1

    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    # Convert mean quaternion back to rotation matrix
    mean_R = R.from_quat(mean_quat).as_matrix()
    mean_t = np.mean(translations, axis=0)

    mean_ext = np.hstack((mean_R, mean_t.reshape(3, 1)))
    avg_extrinsics[cam_idx] = mean_ext

    avg_intrinsics[cam_idx] = np.mean(intrinsics[:, cam_idx], axis=0)


np.save("calibration_images/camera_params/avg_extrinsics.npy", avg_extrinsics)
np.save("calibration_images/camera_params/avg_intrinsics.npy", avg_intrinsics)

print("Averaged extrinsics and intrinsics saved successfully.")
