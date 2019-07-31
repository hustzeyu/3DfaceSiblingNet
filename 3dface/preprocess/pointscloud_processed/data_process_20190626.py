#coding:UTF-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import cv2
import math
import os
import glob
import multiprocessing
from tqdm import tqdm
import img2points
from models.face_class import FaceLandmarks, FaceDetect

def img_norm(img):
    img_max = (img[img != 0]).max()
    img_min = (img[img != 0]).min()
    img_new = (img-img_min)*255.0/(img_max-img_min)
    th = (0-img_min)*255.0/(img_max - img_min)
    img_new[img_new==th] = 0
    img_new = cv2.medianBlur(img_new.astype(np.float32), 5)
    return img_new

def point2gray(x, y, z, alpha, beta, theta, r_list, g_list, b_list):
    u_list, v_list, z_list = [], [], []
    for i, j, k in zip(x, y, z):
        u_list.append((i*616.009)/k)
        v_list.append((j*614.024)/k)
        z_list.append(k*1000)
    width = int(max(u_list) - min(u_list))
    height = int(max(v_list) - min(v_list))
    gray_img = np.zeros((width+1, height+1, 1))
    alpha_img = np.zeros((width+1, height+1, 1))
    beta_img = np.zeros((width+1, height+1, 1))
    theta_img = np.zeros((width+1, height+1, 1))
    rst_img = np.zeros((width+1, height+1, 3))
    r_img = np.zeros((width+1, height+1, 1))
    g_img = np.zeros((width+1, height+1, 1))
    b_img = np.zeros((width+1, height+1, 1))
    color_img = np.zeros((width+1, height+1, 3))
    u_min = min(u_list)
    v_min = min(v_list)
    u_list = [int(i-u_min) for i in u_list]
    v_list = [int(i-v_min) for i in v_list]
    
    for u, v, z, al, be, th, r, g, b in zip(u_list, v_list, z_list, alpha, beta, theta, r_list, g_list, b_list):
        gray_img[u,v] = z
        alpha_img[u,v] = math.acos(abs(al))
        beta_img[u,v] = math.acos(abs(be))
        theta_img[u,v] = math.acos(abs(th))
        r_img[u,v] = int(r)
        g_img[u,v] = int(g)
        b_img[u,v] = int(b)

    img_gray = img_norm(gray_img)
    #cv2.imwrite("gray_me.jpg", img_gray)

    alpha_img = img_norm(alpha_img)
    #cv2.imwrite("alpha_me.jpg", alpha_img)
        
    beta_img = img_norm(beta_img)
    #cv2.imwrite("beta_me.jpg", beta_img)  

    theta_img = img_norm(theta_img)
    #cv2.imwrite("theta_me.jpg", theta_img)

    rst_img[:,:,0] = img_gray
    rst_img[:,:,1] = alpha_img
    rst_img[:,:,2] = theta_img
    #cv2.imwrite("rst_me.jpg", rst_img)

    color_img[:,:,2] = r_img[:,:,0]
    color_img[:,:,1] = g_img[:,:,0]
    color_img[:,:,0] = b_img[:,:,0]
    #cv2.imwrite("color.jpg", color_img)
    return rst_img, color_img


class MyMultiprocess(object):
    def __init__(self, process_num):
        self.pool = multiprocessing.Pool(processes=process_num)

    def work(self, func, args):
        result = []
        for arg in args:
            result.append(self.pool.apply_async(func, (arg,)))
        self.pool.close()
        self.pool.join()
        finally_result = []
        for res in result:
            finally_result.append(res.get())
        return finally_result

def process(img_dir):
    #save_dir = os.path.join("data_processed", img_dir.split("/")[-1])
    save_dir = os.path.join("data_processed_test", img_dir.split("/")[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_list = glob.glob(img_dir + "/*.jpg")
    face_detect = FaceDetect(0)
    face_landmarks = FaceLandmarks(0)
    for color_path in tqdm(img_list):
        rst_list = img2points.main(color_path, face_detect)
        if len(rst_list) == 0:
            continue
        pcd, points_color = rst_list[0], rst_list[1]
        points = np.array(pcd.points)
        x, y, z = points[:,0].tolist(), points[:,1].tolist(), points[:,2].tolist()

        normals = np.array(pcd.normals)
        alpha, beta, theta = normals[:,0].tolist(), normals[:,1].tolist(), normals[:,2].tolist()

        b_list, g_list, r_list = points_color[:,3].tolist(), points_color[:, 4].tolist(), points_color[:,5]
        rst_img, color_img = point2gray(x,y,z, alpha, beta, theta, r_list, g_list, b_list)

        if color_img is None or not color_img.any():
            continue
        #landmarks = face_landmarks.get_points(color_img)
        #rst_align_img = face_landmarks.alignment(rst_img, landmarks)
        #color_align_img = face_landmarks.alignment(color_img, landmarks)
        img_name = os.path.basename(color_path)
        color_save_path = os.path.join(save_dir, img_name)
        d3_save_path = color_save_path.replace("color", "3d")
        finall_img = np.hstack([rst_img, color_img])
        cv2.imwrite(color_save_path, finall_img)
        #cv2.imwrite(d3_save_path, rst_align_img)
    return None
   
if __name__ == "__main__":
    #root_dir = "/home/gp/work/project/kaoqing/dataSet_G/"
    root_dir = "/home/gp/work/project/3d_face/datas/get_data_tool/face_det/dataset/"
    img_dir = os.listdir(root_dir)
    img_dir_list = [os.path.join(root_dir, i) for i in img_dir]
    mymultiprocess = MyMultiprocess(3)
    rst = mymultiprocess.work(process, img_dir_list)
    #for img_dir in img_dir_list:
    #    process(img_dir)
    





