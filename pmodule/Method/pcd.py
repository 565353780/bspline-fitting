import numpy as np
import open3d as o3d


def getPointCloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def downSample(pcd, sample_point_num):
    if sample_point_num < 1:
        print("[WARN][pcd::downSample]")
        print("\t sample_point_num < 1!")
        return None

    return pcd.farthest_point_down_sample(sample_point_num)
