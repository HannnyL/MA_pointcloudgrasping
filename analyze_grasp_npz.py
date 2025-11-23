import os
from datetime import datetime
import numpy as np
from grasp_store import GraspStore, GraspData
import open3d as o3d



# mean pairwise Euclidean distance

def mean_pairwise_distance(points):
    n = len(points)
    if n < 2:
        raise ValueError("At least 2 points are required to compute pairwise distance.")

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            dists.append(d)

    return float(np.mean(dists))



def compute_topk_mean_distance(npz_file, k):
    logs = []

    def log(msg: str):
        print(msg)
        logs.append(msg)

    log("=== Function 1: top-k mean distance ===")
    log(f"File: {npz_file}")

    if not os.path.isfile(npz_file):
        log("Error: file does not exist.")
        # Still try to save the log
        dirpath = os.path.dirname(npz_file) or "."
        log_path = os.path.join(dirpath, "top_k_position_deviation.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    store = GraspStore.load_npz(npz_file)
    topk = store.top_k(k)

    if len(topk) < 2:
        log(f"Not enough points (got {len(topk)}), cannot compute mean distance.")
        # Save log anyway
        dirpath = os.path.dirname(npz_file) or "."
        log_path = os.path.join(dirpath, "top_k_position_deviation.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    pts = np.array([g.TCP[:3] for g in topk], dtype=float)
    mean_dist = mean_pairwise_distance(pts)

    log(f"Requested k = {k}, actually used top-{len(topk)} points.")
    log(f"Mean pairwise Euclidean distance of xyz: [[{mean_dist:.6f}]]")

    # Save logs to txt in the same directory as the npz file
    dirpath = os.path.dirname(npz_file) or "."
    # File name as requested: top_k_position_deviation (txt file)
    log_path = os.path.join(dirpath, "top_k_position_deviation.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    log(f"Result has been saved to: {log_path}")
    return mean_dist


def compute_best_mean_distance(npz_file_list):
    logs = []

    def log(msg: str):
        print(msg)
        logs.append(msg)

    log("=== Function 2: mean distance of best grasp of each file ===")
    log("Input files:")
    for f in npz_file_list:
        log(f"  - {f}")

    best_points = []

    for f in npz_file_list:
        if not os.path.isfile(f):
            log(f"Warning: file does not exist, skipped: {f}")
            continue

        store = GraspStore.load_npz(f)
        best = store.best()
        if best is None:
            log(f"Warning: no grasp data in file, skipped: {f}")
            continue

        best_points.append(np.array(best.TCP[:3], dtype=float))

    if len(best_points) < 2:
        log(f"Not enough valid best points (got {len(best_points)}), cannot compute mean distance.")

        # Even in this case, we still save the log files to each directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirs = set(os.path.dirname(f) or "." for f in npz_file_list)
        for d in dirs:
            log_path = os.path.join(d, f"best_position_deviation_{timestamp}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(logs))
            log(f"Log has been saved to: {log_path}")
        return None

    pts = np.array(best_points, dtype=float)
    mean_dist = mean_pairwise_distance(pts)

    log(f"Number of files with valid best grasps: {len(best_points)}")
    log(f"Mean pairwise Euclidean distance of xyz among best points: [[{mean_dist:.6f}]]")

    # Save logs to txt in each file directory (avoid overwrite by using timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = set(os.path.dirname(f) or "." for f in npz_file_list)

    for d in dirs:
        log_path = os.path.join(d, f"repeatability_deviation_best_position_{timestamp}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Result has been saved to: {log_path}")

    return mean_dist


def compute_accuracy_deviation_with_pcd(npz_file, pcd_file):
    logs = []

    def log(msg: str):
        print(msg)
        logs.append(msg)

    log("=== Function 3: accuracy deviation with PCD and best grasp ===")
    log(f"NPZ file: {npz_file}")
    log(f"PCD file: {pcd_file}")

    dirpath = os.path.dirname(npz_file) or "."
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(dirpath, f"accuracy_deviation_best_position_{timestamp}.txt")

    # Check open3d
    if o3d is None:
        log("Error: open3d is not installed. Please install it with 'pip install open3d'.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    # Check files
    if not os.path.isfile(npz_file):
        log("Error: NPZ file does not exist.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    if not os.path.isfile(pcd_file):
        log("Error: PCD file does not exist.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    # Load PCD
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    if points.size == 0:
        log("Error: PCD has no points.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    mean_point = points.mean(axis=0)
    log(f"Mean point of PCD: {mean_point.tolist()}")

    obb = pcd.get_oriented_bounding_box()
    obb_center = np.asarray(obb.center)
    log(f"OBB center point: {obb_center.tolist()}")

    # Load best grasp from npz
    store = GraspStore.load_npz(npz_file)
    best = store.best()
    if best is None:
        log("Error: no grasp data in NPZ file, cannot get best grasp.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        log(f"Log has been saved to: {log_path}")
        return None

    best_point = np.array(best.TCP[:3], dtype=float)
    log(f"Best grasp point from NPZ (xyz): {best_point.tolist()}")

    # Distances
    dist_best_mean = float(np.linalg.norm(best_point - mean_point))
    dist_best_obb = float(np.linalg.norm(best_point - obb_center))
    dist_mean_obb = float(np.linalg.norm(mean_point - obb_center))

    log("Pairwise distances (in the same unit as the coordinates):")
    log(f"  Distance between BEST and MEAN: {dist_best_mean:.6f}")
    log(f"  Distance between BEST and OBB_CENTER: {dist_best_obb:.6f}")
    log(f"  Distance between MEAN and OBB_CENTER (reference): {dist_mean_obb:.6f}")

    # Save to txt
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    log(f"Result has been saved to: {log_path}")

    # Return a small dictionary if you want to reuse programmatically
    return {
        "best_mean": dist_best_mean,
        "best_obb_center": dist_best_obb,
        "mean_obb_center": dist_mean_obb,
    }





if __name__ == "__main__":

    # path for top_k and best accuracy annlysis
    FILE_PATH = r"results/20251113_122606-Franka-foam_brick_100K-0.1/grasp_data/20251113_122606-Franka-foam_brick_100K.npz"
    K = 5

    PCD_PATH = r"object/validation_object/01_YCB_061_foam_brick/foam_brick_100K.pcd"

    # path for best repeatability consistency annalysis
    FILE_LIST = [
        r"results/20251114_085352-Franka-foam_brick_1M-0.1/grasp_data/20251114_085352-Franka-foam_brick_1M-0.1.npz",
        r"results/20251114_083829-Franka-foam_brick_1K-0.1/grasp_data/20251114_083829-Franka-foam_brick_1K-0.1.npz",
        r"results/20251114_084935-Franka-foam_brick_5K-0.1/grasp_data/20251114_084935-Franka-foam_brick_5K-0.1.npz",
        r"results/20251114_085221-Franka-foam_brick_10K-0.1/grasp_data/20251114_085221-Franka-foam_brick_10K-0.1.npz",
        r"results/20251113_122606-Franka-foam_brick_100K-0.1/grasp_data/20251113_122606-Franka-foam_brick_100K.npz"
    ]

    choice = '3' # '1' = top_k_compare (single npz file) mode or '2' = repeatability_deviation_best_position (multiple npz file)mode or '3' = accuracy_deviation_best_position (single npz file and pcd file) mode

    if choice == "1":
        compute_topk_mean_distance(FILE_PATH, K)
    elif choice == "2":
        compute_best_mean_distance(FILE_LIST)
    elif choice == "3":
        compute_accuracy_deviation_with_pcd(FILE_PATH, PCD_PATH)
    else:
        print("Invalid input. No function was executed.")
