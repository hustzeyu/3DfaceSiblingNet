#coding:UTF-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import cv2
import math
from utils import obj_reader


def img_norm(img):
    img_max = (img[img != 0]).max()
    img_min = (img[img != 0]).min()
    img_new = (img-img_min)*255.0/(img_max-img_min)
    th = (0-img_min)*255.0/(img_max - img_min)
    img_new[img_new==th] = 0
    img_new = cv2.medianBlur(img_new.astype(np.float32), 3)
    return img_new


def point2gray(x, y, z, alpha, beta, theta):
    u_list, v_list, z_list = [], [], []
    for i, j, k in zip(x, y, z):
        u_list.append((i*616.009)*0.1/k)
        v_list.append((j*614.024)*0.1/k)
        z_list.append(k/1000)
    width = int(max(u_list) - min(u_list))
    height = int(max(v_list) - min(v_list))
    gray_img = np.zeros((width+1, height+1, 1))
    alpha_img = np.zeros((width+1, height+1, 1))
    beta_img = np.zeros((width+1, height+1, 1))
    theta_img = np.zeros((width+1, height+1, 1))
    rst_img = np.zeros((width+1, height+1, 3))
    u_min = min(u_list)
    v_min = min(v_list)
    u_list = [int(i-u_min) for i in u_list]
    v_list = [int(i-v_min) for i in v_list]
    #import pdb;pdb.set_trace()
    
    for u, v, z, al, be, th in zip(u_list, v_list, z_list, alpha, beta, theta):
        gray_img[u,v] = z
        alpha_img[u,v] = math.acos(al)
        beta_img[u,v] = math.acos(be)
        theta_img[u,v] = math.acos(th)

    img_gray = img_norm(gray_img)
    cv2.imwrite("gray_me.jpg", img_gray)

    alpha_img = img_norm(alpha_img)
    cv2.imwrite("alpha_me.jpg", alpha_img)
        
    beta_img = img_norm(beta_img)
    cv2.imwrite("beta_me.jpg", beta_img)  

    theta_img = img_norm(theta_img)
    cv2.imwrite("theta_me.jpg", theta_img)

    rst_img[:,:,0] = img_gray
    rst_img[:,:,1] = alpha_img
    rst_img[:,:,2] = theta_img
    cv2.imwrite("rst_me.jpg", rst_img)


if __name__ == "__main__":
    x, y, z, alpha, beta, theta = obj_reader("30.obj")
    point2gray(x,y,z, alpha, beta, theta)
