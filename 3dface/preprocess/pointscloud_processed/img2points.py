#coding:UTF-8
import cv2
import numpy as np
import os
import math
from tqdm import tqdm
import open3d as o3d
from models.face_class import FaceDetect

def img2points(color_img, depth_img, box):
    cX = 320.351
    cY = 232.881
    fx = 616.009
    fy = 614.024
    w = 640
    h = 480
    scalingFactor = 1000
    points = []
    v_mean = int((box[0]+box[2])*0.5)
    u_mean = int((box[1]+box[3])*0.5)
    z_mean = depth_img[u_mean, v_mean] * 1.0 / scalingFactor
    for v in range(box[0], box[2]):
        for u in range(box[1], box[3]):
            color = color_img[u, v,(2,1,0)]  # bgr
            Z = depth_img[u, v] * 1.0/ scalingFactor
            if Z == 0 or abs(Z-z_mean)>0.15:
                continue
            X = (u - cX) * Z / fx
            Y = (v - cY) * Z / fy
            points.append([X, Y, Z, color[0], color[1], color[2]])
    return np.array(points)

def path2points(color_path, face_detect):
    #depth_path = color_path.replace("color", "depth").replace("jpg", "png")
    depth_path = color_path.replace(".jpg", ".npy")
    color_img = cv2.imread(color_path)
    #depth_img = cv2.imread(depth_path, -1)
    depth_img = np.load(depth_path)
    if color_img is None or not color_img.any():
        return []
    if depth_img is None or not depth_img.any():
        return []
    faceboxes = face_detect.getboxes(color_img)
    if not faceboxes:
        return []
    face_box = face_detect.get_big_box(faceboxes)
    face_box = face_detect.shrink_box(face_box, 0.15)
    points = img2points(color_img, depth_img, face_box)
    return points


def get_normals(pcd):
    #print("Downsample the point cloud with a voxel of 0.05")
    #downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=0.0005)
    #o3d.visualization.draw_geometries([downpcd])

    #print("Recompute the normal of the downsampled point cloud")
    o3d.geometry.estimate_normals(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))
    #o3d.visualization.draw_geometries([downpcd])
    return pcd

def main(color_path, face_detect):
    points = path2points(color_path, face_detect)
    if len(points) == 0:
        return []
    xyz = points[:, 0:3]
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    down_pcd = get_normals(pcd)
    #o3d.io.write_point_cloud("sync.ply", pcd)
    # Load saved point cloud and visualize it
    #pcd_load = o3d.io.read_point_cloud("sync.ply")
    #o3d.visualization.draw_geometries([pcd_load])
    return [down_pcd, points]


if __name__ == "__main__":
    color_path = "/home/gp/work/project/kaoqing/dataSet_G/anony_3/color_229.jpg"
    face_detect = FaceDetect(0)
    pcd, points = main(color_path)

