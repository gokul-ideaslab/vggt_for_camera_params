import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

def project_to_valid_rotation(R):
    """
    Convert a near-rotation or reflected matrix into the closest valid
    right-handed rotation matrix with det(R)=+1.
    """
    U, _, Vt = np.linalg.svd(R)
    R_valid = U @ Vt

    if np.linalg.det(R_valid) < 0:
        U[:, -1] *= -1
        R_valid = U @ Vt

    return R_valid


def fix_extrinsic_rotations(extrinsics):
    """
    extrinsics: (C, 3, 4)
    """
    fixed = extrinsics.copy()

    for c in range(extrinsics.shape[0]):
        R = fixed[c, :, :3]
        t = fixed[c, :, 3]

        R_fixed = project_to_valid_rotation(R)

        fixed[c, :, :3] = R_fixed
        fixed[c, :, 3] = t

        print(
            f"cam {c}: det before={np.linalg.det(R):.6f}, "
            f"det after={np.linalg.det(R_fixed):.6f}"
        )

    return fixed

def rotate_world_180(extrinsics, axis="y"):
    """
    Apply a proper 180-degree world rotation.
    This keeps det(R)=+1.
    """
    if axis == "x":
        A = np.diag([1, -1, -1, 1])
    elif axis == "y":
        A = np.diag([-1, 1, -1, 1])
    elif axis == "z":
        A = np.diag([-1, -1, 1, 1])
    else:
        raise ValueError("axis must be x, y, or z")

    out = np.zeros_like(extrinsics)

    for c in range(extrinsics.shape[0]):
        E = np.eye(4)
        E[:3, :4] = extrinsics[c]

        E_new = E @ A
        out[c] = E_new[:3, :4]

    return out

# Projection
def project_points(X, K, R, t):
    """
    X: (N, 3)
    K: (3, 3)
    R: (3, 3)
    t: (3,)
    """
    X_cam = (R @ X.T).T + t

    z = X_cam[:, 2:3]
    z = np.clip(z, 1e-6, None)

    x_norm = X_cam[:, :2] / z
    x_pix = (K[:2, :2] @ x_norm.T).T + K[:2, 2]

    return x_pix

# Linear triangulation
def triangulate_point(obs_2d, Ps):
    """
    obs_2d: list of (x, y)
    Ps: list of projection matrices K[R|t]
    """
    A = []

    for (x, y), P in zip(obs_2d, Ps):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]

    return X


def triangulate_all_keypoints(keypoints_2d, confidences, K_all, extrinsics, conf_thresh=0.3):
    """
    keypoints_2d: (T, C, J, 2)
    confidences: (T, C, J)
    K_all: (C, 3, 3)
    extrinsics: (C, 3, 4)
    """
    T, C, J, _ = keypoints_2d.shape

    Ps = []
    for c in range(C):
        P = K_all[c] @ extrinsics[c]
        Ps.append(P)

    X_init = np.zeros((T, J, 3), dtype=np.float64)

    for t in range(T):
        for j in range(J):
            obs = []
            valid_Ps = []

            for c in range(C):
                if confidences[t, c, j] > conf_thresh:
                    obs.append(keypoints_2d[t, c, j])
                    valid_Ps.append(Ps[c])

            if len(obs) >= 2:
                X_init[t, j] = triangulate_point(obs, valid_Ps)
            else:
                X_init[t, j] = np.nan

    # Fill missing 3D points with per-joint median
    for j in range(J):
        valid = ~np.isnan(X_init[:, j, 0])
        if valid.sum() > 0:
            median_val = np.nanmedian(X_init[:, j], axis=0)
            X_init[~valid, j] = median_val
        else:
            X_init[:, j] = 0.0

    return X_init

def pack_params(rotvecs, translations, X):
    return np.concatenate([
        rotvecs.ravel(),
        translations.ravel(),
        X.ravel(),
    ])


def unpack_params(params, num_cams, num_frames, num_joints):
    num_opt_cams = num_cams - 1

    idx = 0

    rotvecs = params[idx:idx + num_opt_cams * 3].reshape(num_opt_cams, 3)
    idx += num_opt_cams * 3

    translations = params[idx:idx + num_opt_cams * 3].reshape(num_opt_cams, 3)
    idx += num_opt_cams * 3

    X = params[idx:].reshape(num_frames, num_joints, 3)

    return rotvecs, translations, X


def ba_residuals(
    params,
    keypoints_2d,
    confidences,
    K_all,
    R_ref,
    t_ref,
    conf_thresh=0.3,
    temporal_weight=0.01,
):
    """
    Cam1 is fixed.
    Cam2 and cam3 are optimized.
    Intrinsics are fixed.
    """
    T, C, J, _ = keypoints_2d.shape

    rotvecs, translations, X = unpack_params(params, C, T, J)

    R_all = [R_ref]
    t_all = [t_ref]

    for i in range(C - 1):
        R_all.append(Rot.from_rotvec(rotvecs[i]).as_matrix())
        t_all.append(translations[i])

    residuals = []

    # Reprojection residuals
    for t in range(T):
        for c in range(C):
            valid = confidences[t, c] > conf_thresh

            if valid.sum() == 0:
                continue

            X_valid = X[t, valid]
            x_obs = keypoints_2d[t, c, valid]

            x_proj = project_points(
                X_valid,
                K_all[c],
                R_all[c],
                t_all[c],
            )

            w = np.sqrt(confidences[t, c, valid])[:, None]
            reproj_res = (x_proj - x_obs) * w

            residuals.append(reproj_res.ravel())

    # Temporal smoothness residual
    if temporal_weight > 0 and T > 1:
        temporal_res = (X[1:] - X[:-1]) * temporal_weight
        residuals.append(temporal_res.ravel())

    return np.concatenate(residuals)

if __name__ == "__main__":
    K_all = np.load("../calibration_images/winston/camera_params/avg_intrinsics.npy")       # (3, 3, 3)
    extrinsics = np.load("../calibration_images/winston/camera_params/avg_extrinsics.npy")  # (3, 3, 4)
    # extrinsics = rotate_world_180(extrinsics, axis="y")
    # extrinsics = fix_extrinsic_rotations(extrinsics)

    fo = np.load("/home/gokul/ideaslab/yolo_model/kps/winston/golf/fo.npy")
    dtl = np.load("/home/gokul/ideaslab/yolo_model/kps/winston/golf/dtl.npy")
    rear = np.load("/home/gokul/ideaslab/yolo_model/kps/winston/golf/rear.npy")

    print("FO shape:", fo.shape, "DTL shape:", dtl.shape, "Rear shape:", rear.shape)

    fo = fo[:266, :, :]
    dtl = dtl[:266, :, :]
    rear = rear[:266, :, :]
    all_kps = np.stack((fo, dtl, rear), axis = 1)
    
    keypoints_2d = all_kps[..., :2]   # (T, 3, J, 2)
    confidences = all_kps[..., 2]     # (T, 3, J)

    frame_ids = np.linspace(0, 265, 40).astype(int)
    keypoints_2d_ba = keypoints_2d[frame_ids].astype(np.float64)
    confidences_ba = confidences[frame_ids].astype(np.float64)

    K_all = K_all.astype(np.float64)
    extrinsics = extrinsics.astype(np.float64)

    T, C, J, _ = keypoints_2d_ba.shape

    print("Keypoints_ba:", keypoints_2d.shape)
    print("Confidences_ba:", confidences.shape)
    print("Intrinsics:", K_all.shape)
    print("Extrinsics:", extrinsics.shape)

    assert C == 3, "This script expects 3 cameras."

    print("Triangulating initial 3D joints...")

    X_init = triangulate_all_keypoints(
        keypoints_2d_ba,
        confidences_ba,
        K_all,
        extrinsics,
        conf_thresh=0.3,
    )

    print("Initial 3D shape:", X_init.shape)

    # Fix cam1 as reference
    R_ref = extrinsics[0, :, :3]
    t_ref = extrinsics[0, :, 3]

    # Optimize cam2 and cam3
    rotvecs_init = []
    translations_init = []

    for cam_idx in range(1, C):
        R_init = extrinsics[cam_idx, :, :3]
        t_init = extrinsics[cam_idx, :, 3]

        rotvecs_init.append(Rot.from_matrix(R_init).as_rotvec())
        translations_init.append(t_init)

    rotvecs_init = np.asarray(rotvecs_init)
    translations_init = np.asarray(translations_init)

    params0 = pack_params(
        rotvecs_init,
        translations_init,
        X_init,
    )

    print("Running bundle adjustment...")

    result = least_squares(
        ba_residuals,
        params0,
        args=(
            keypoints_2d_ba,
            confidences_ba,
            K_all,
            R_ref,
            t_ref,
        ),
        kwargs={
            "conf_thresh": 0.3,
            "temporal_weight": 0.01,
        },
        loss="soft_l1",
        f_scale=10.0,
        max_nfev=100,
        verbose=2,
        tr_solver="lsmr",
        x_scale="jac",
    )

    print("BA success:", result.success)
    print("Final cost:", result.cost)

    rotvecs_opt, translations_opt, X_opt = unpack_params(result.x, C, T, J)

    final_extrinsics = np.zeros((C, 3, 4), dtype=np.float64)

    # cam1 fixed
    final_extrinsics[0, :, :3] = R_ref
    final_extrinsics[0, :, 3] = t_ref

    # cam2/cam3 optimized
    for i in range(C - 1):
        cam_idx = i + 1

        final_extrinsics[cam_idx, :, :3] = Rot.from_rotvec(
            rotvecs_opt[i]
        ).as_matrix()

        final_extrinsics[cam_idx, :, 3] = translations_opt[i]

    X_all = triangulate_all_keypoints(
        keypoints_2d,
        confidences,
        K_all,
        final_extrinsics,
        conf_thresh=0.3,
    )

    print("Refined 3D shape:", X_all.shape)
    
    np.save("../calibration_images/winston/ba_refined_extrinsics.npy", final_extrinsics)
    np.save("../calibration_images/winston/ba_refined_3d_joints.npy", X_all)
    np.save("../calibration_images/winston/ba_result_cost.npy", np.array([result.cost]))

    print("Saved:")
    print("ba_refined_extrinsics.npy", final_extrinsics.shape)
    print("ba_refined_3d_joints.npy", X_all.shape)