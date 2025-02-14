import numpy as np
import open3d as o3d
import pytorch3d
from pytorch3d.io import load_ply


def normalize_ply(pcd):


    points = np.asarray(pcd.points)

    center = np.mean(points, axis=0)
    
    max_distance = np.max(np.linalg.norm(points - center, axis=1))

    normalized_points = (points - center) / max_distance

    pcd.points = o3d.utility.Vector3dVector(normalized_points)


    return pcd, center, max_distance




def normalize_ply_file(input_file):

    ply=o3d.io.read_point_cloud(input_file)

    normalized_ply, normal_center, normal_scale = normalize_ply(ply)

    return normalized_ply, normal_center, normal_scale