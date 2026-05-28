import numpy as np

def extract_params():
    intr = np.load("calibration_images/camera_params/avg_intrinsics.npy")
    extr = np.load("calibration_images/camera_params/avg_extrinsics.npy")

    cams_t = extr[:, :, :3]
    cams_r = extr[:, :, 3]

    return intr[0], cams_t[0], cams_r[0], intr[1], cams_t[1], cams_r[1], intr[2], cams_t[2], cams_r[2]

def make_projection_matrix(K, R_wc, t_wc):
    """
    P = K [R|t] for world->camera
    K: intrinsic matrix (3x3)
    R_wc: rotation matrix (3x3) for world->camera
    t_wc: translation vector (3,) for world->camera
    """
    K = np.asarray(K, dtype=np.float64)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    t_wc = np.asarray(t_wc, dtype=np.float64)

    Rt = np.hstack([R_wc, t_wc.reshape(3, 1)])
    P = K @ Rt
    return P, K, R_wc, t_wc

# Triangulation (DLT)
def triangulate_point_multi_view(P_list, x_list):
    """
    Multi-view DLT triangulation for a single point.
    P_list: list of (3x4) projection matrices (N views)
    x_list: list of (2,) pixel coords (same length as P_list)

    returns:
      X (3,) world coords
      success (bool)
    """
    assert len(P_list) == len(x_list)
    n = len(P_list)
    if n < 2:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), False

    A = []
    for P, x in zip(P_list, x_list):
        u, v = float(x[0]), float(x[1])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A, axis=0)  # (2n, 4)

    # Solve A X = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if np.isclose(X_h[3], 0.0):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), False
    X = X_h[:3] / X_h[3]
    return X.astype(np.float64), True

def project_point(P, X):
    """
    Project 3D point X (3,) with projection matrix P (3x4) -> (2,)
    """
    X_h = np.append(X, 1.0)
    x = P @ X_h
    if np.isclose(x[2], 0.0):
        return np.array([np.nan, np.nan], dtype=np.float64)
    return (x[:2] / x[2]).astype(np.float64)

# 3-camera triangulation
def triangulate_sequence(
    kps_fo, kps_dtl, kps_rear,
    fo_intr_params, fo_R, fo_t,
    dtl_intr_params, dtl_R, dtl_t,
    rear_intr_params, rear_R, rear_t,
    invalid_value=0.0,
    min_valid_pixels=1.0
):
    """
    kps_*: (F, J, 2) pixel coords for each camera (fo, dtl, rear)
    intr_params: K matrix (3x3) intrinsic matrix
    R_*: rotation matrix (3x3) for world->camera
    t_*: translation vector (3,) for world->camera

    Returns:
      X_world: (F, J, 3)
      reproj_err: (F, J) average reprojection error across used cameras
      valid_mask: (F, J) True if triangulated (>=2 views valid)
    """
    kps_fo = np.asarray(kps_fo, dtype=np.float64)
    kps_dtl = np.asarray(kps_dtl, dtype=np.float64)
    kps_rear = np.asarray(kps_rear, dtype=np.float64)

    if kps_fo.shape != kps_dtl.shape or kps_fo.shape != kps_rear.shape:
        raise ValueError(f"Keypoint shapes must match. "
                         f"fo={kps_fo.shape}, dtl={kps_dtl.shape}, rear={kps_rear.shape}")

    F, J, D = kps_fo.shape
    if D != 2:
        raise ValueError("Keypoints must be (frames, joints, 2)")

    # Build projection matrices
    P_fo, _, _, _ = make_projection_matrix(fo_intr_params, fo_R, fo_t)
    P_dtl, _, _, _ = make_projection_matrix(dtl_intr_params, dtl_R, dtl_t)
    P_rear, _, _, _ = make_projection_matrix(rear_intr_params, rear_R, rear_t)

    P_list_all = [P_fo, P_dtl, P_rear]

    # Validity check per keypoint
    def valid_xy(xy):
        u, v = xy[..., 0], xy[..., 1]
        bad = ~np.isfinite(u) | ~np.isfinite(v)
        if invalid_value is not None:
            bad |= ((u == invalid_value) & (v == invalid_value))
        bad |= (u < min_valid_pixels) | (v < min_valid_pixels)
        return ~bad

    v_fo = valid_xy(kps_fo)
    v_dtl = valid_xy(kps_dtl)
    v_rear = valid_xy(kps_rear)

    X_world = np.full((F, J, 3), np.nan, dtype=np.float64)
    reproj_err = np.full((F, J), np.nan, dtype=np.float64)
    valid_mask = np.zeros((F, J), dtype=bool)

    for f in range(F):
        for j in range(J):
            views = []
            P_use = []
            if v_fo[f, j]:
                P_use.append(P_fo)
                views.append(kps_fo[f, j])
            if v_dtl[f, j]:
                P_use.append(P_dtl)
                views.append(kps_dtl[f, j])
            if v_rear[f, j]:
                P_use.append(P_rear)
                views.append(kps_rear[f, j])

            if len(P_use) < 2:
                continue

            X, ok = triangulate_point_multi_view(P_use, views)
            if not ok:
                continue

            # Compute reprojection error (mean over used cameras)
            errs = []
            for P, x_obs in zip(P_use, views):
                x_hat = project_point(P, X)
                if np.any(~np.isfinite(x_hat)):
                    continue
                errs.append(np.linalg.norm(x_hat - x_obs))
            if len(errs) == 0:
                continue

            X_world[f, j] = X
            reproj_err[f, j] = float(np.mean(errs))
            valid_mask[f, j] = True

    return X_world, reproj_err, valid_mask


def get_3d_keypoints(keypoints_fo, keypoints_dtl, keypoints_rear):
    fo_K, fo_R, fo_t, dtl_K, dtl_R, dtl_t, rear_K, rear_R, rear_t = extract_params()
    
    # Perform triangulation with extracted parameters
    X_3d, reproj_error, valid = triangulate_sequence(
        keypoints_fo, keypoints_dtl, keypoints_rear,
        fo_K, fo_R, fo_t,
        dtl_K, dtl_R, dtl_t,
        rear_K, rear_R, rear_t,
        invalid_value=0.0,
        min_valid_pixels=1.0
    )

    print("3D Points shape:", X_3d.shape)
    print("Reprojection Error:", np.nanmean(reproj_error))
    print("Valid: ", valid.mean())

    np.save("resonstructed_3d_points.npy", X_3d)

    return X_3d, reproj_error

if __name__ == "__main__":
    fo = np.load("/home/gokul/ideaslab/yolo_model/kps/geenral_pose/rear.npy")
    dtl = np.load("/home/gokul/ideaslab/yolo_model/kps/geenral_pose/dtl.npy")
    rear = np.load("/home/gokul/ideaslab/yolo_model/kps/geenral_pose/fo.npy")

    fo = fo[:, :, :2]
    dtl = dtl[:1952, :, :2]
    rear = rear[:1952, :, :2]

    X_3d, reproj_error = get_3d_keypoints(fo, dtl, rear)

    print(f"Reprojecton error (mean): {reproj_error}")

    np.save("3d_keypoints.npy", X_3d)