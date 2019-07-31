# coding:UTF-8
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import cv2
import os
import math

flag = 0


def wrl_reader(obj_file):
    alpha, beta, theta, x, y, z, r, g, b = [], [], [], [], [], [], [], [], []
    with open(obj_file, "r") as f:
        lines = f.readlines()
        flag = 0

        for line in lines:

            if line.find('point [') > 0:
                flag = 1
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip("point [")
                line = line.strip(",")
                parts = line.split(" ")
                x.append(float(parts[0]))
                y.append(float(parts[1]))
                z.append(float(parts[2]))
            elif (flag == 1) & (line.find(']') < 0):
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip(",")
                line = line.strip('')
                parts = line.split(" ")
                x.append(float(parts[1]))
                y.append(float(parts[2]))
                z.append(float(parts[3]))
            elif (line.find(']') > 0) & (flag == 1):
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip("]")
                line = line.strip(",")
                parts = line.split(" ")
                x.append(float(parts[1]))
                y.append(float(parts[2]))
                z.append(float(parts[3]))
                flag = 0

            elif line.find('vector [') > 0:
                flag = 2
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip("vector [")
                line = line.strip(",")
                parts = line.split(" ")
                alpha.append(float(parts[0]))
                beta.append(float(parts[1]))
                theta.append(float(parts[2]))
            elif (flag == 2) & (line.find(']') < 0):
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip(",")
                parts = line.split(" ")
                alpha.append(float(parts[1]))
                beta.append(float(parts[2]))
                theta.append(float(parts[3]))
            elif (line.find(']') > 0) & (flag == 2):
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip("]")
                line = line.strip(",")
                parts = line.split(" ")
                alpha.append(float(parts[1]))
                beta.append(float(parts[2]))
                theta.append(float(parts[3]))
                flag = 0

            elif line.find('Color { color [') > 0:
                flag = 3
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip("Color { color [  ")
                line = line.strip(",")
                parts = line.split(" ")
                r.append(float(parts[0]))
                g.append(float(parts[1]))
                b.append(float(parts[2]))
            elif (flag == 3) & (line.find(']') < 0):
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip(",")
                parts = line.split(" ")
                r.append(float(parts[1]))
                g.append(float(parts[2]))
                b.append(float(parts[3]))
            elif (line.find(']') > 0) & (flag == 3):
                line = line.strip("\r\n")
                line = line.strip("\t")
                line = line.strip("]")
                line = line.strip(",")
                parts = line.split(" ")
                r.append(float(parts[1]))
                g.append(float(parts[2]))
                b.append(float(parts[3]))
                flag = 0

        return x, y, z, alpha, beta, theta, r, g, b


def img_norm(img):
    img_max = (img[img != 0]).max()
    img_min = (img[img != 0]).min()
    img_new = (img - img_min) * 255.0 / (img_max - img_min)
    th = (0 - img_min) * 255.0 / (img_max - img_min)
    img_new[img_new == th] = 0
    img_new = cv2.medianBlur(img_new.astype(np.float32), 3)
    return img_new


def point2gray(x, y, z, alpha, beta, theta, r, g, b, dst_folder, dir, num):
    u_list, v_list, z_list = [], [], []
    for i, j, k in zip(x, y, z):
        u_list.append((i * 616.009) / k)
        v_list.append((j * 614.024) / k)
        z_list.append(k / 1000)
    width = int(max(u_list) - min(u_list))
    height = int(max(v_list) - min(v_list))
    gray_img = np.zeros((width + 1, height + 1, 1))
    alpha_img = np.zeros((width + 1, height + 1, 1))
    beta_img = np.zeros((width + 1, height + 1, 1))
    theta_img = np.zeros((width + 1, height + 1, 1))

    r_img = np.zeros((width + 1, height + 1, 1))
    g_img = np.zeros((width + 1, height + 1, 1))
    b_img = np.zeros((width + 1, height + 1, 1))

    rst_img = np.zeros((width + 1, height + 1, 3))
    rgb_img = np.zeros((width + 1, height + 1, 3))
    u_min = min(u_list)
    v_min = min(v_list)
    u_list = [int(i - u_min) for i in u_list]
    v_list = [int(i - v_min) for i in v_list]

    for u, v, z, al, be, th, rr, gg, bb in zip(u_list, v_list, z_list, alpha, beta, theta, r, g, b):
        gray_img[u, v] = z
        # al,be,th are angles
        alpha_img[u, v] = abs(al)
        beta_img[u, v] = abs(be)
        theta_img[u, v] = abs(th)

        r_img[u, v] = rr
        g_img[u, v] = gg
        b_img[u, v] = bb

    img_gray = img_norm(gray_img)
    dstfolder1 = os.path.join(dst_folder, "gray")
    graypath = os.path.join(dstfolder1, dir)
    if not os.path.exists(graypath):
        os.mkdir(graypath)
    graypath = os.path.join(graypath, num + '.jpg')
    cv2.imwrite(graypath, img_gray)

    alpha_img = img_norm(alpha_img)
    dstfolder2 = os.path.join(dst_folder, "alpha")
    alphapath = os.path.join(dstfolder2, dir)
    if not os.path.exists(alphapath):
        os.mkdir(alphapath)
    alphapath = os.path.join(alphapath, num + '.jpg')
    cv2.imwrite(alphapath, alpha_img)

    beta_img = img_norm(beta_img)
    dstfolder3 = os.path.join(dst_folder, "beta")
    betapath = os.path.join(dstfolder3, dir)
    if not os.path.exists(betapath):
        os.mkdir(betapath)
    betapath = os.path.join(betapath, num + '.jpg')
    cv2.imwrite(betapath, beta_img)

    theta_img = img_norm(theta_img)
    dstfolder4 = os.path.join(dst_folder, "theta")
    thetapath = os.path.join(dstfolder4, dir)
    if not os.path.exists(thetapath):
        os.mkdir(thetapath)
    thetapath = os.path.join(thetapath, num + '.jpg')
    cv2.imwrite(thetapath, theta_img)

    rst_img[:, :, 0] = img_gray
    rst_img[:, :, 1] = alpha_img
    rst_img[:, :, 2] = theta_img
    dstfolder5 = os.path.join(dst_folder, "rst")
    rstpath = os.path.join(dstfolder5, dir)
    if not os.path.exists(rstpath):
        os.mkdir(rstpath)
    rstpath = os.path.join(rstpath, num + '.jpg')
    cv2.imwrite(rstpath, rst_img)

    r_img = img_norm(r_img)
    b_img = img_norm(b_img)
    g_img = img_norm(g_img)
    rgb_img[:, :, 2] = r_img
    rgb_img[:, :, 0] = b_img
    rgb_img[:, :, 1] = g_img
    dstfolder6 = os.path.join(dst_folder, "rgb")
    rgbpath = os.path.join(dstfolder6, dir)
    if not os.path.exists(rgbpath):
        os.mkdir(rgbpath)
    rgbpath = os.path.join(rgbpath, num + '.jpg')
    cv2.imwrite(rgbpath, rgb_img)


if __name__ == "__main__":

    wrl_folder = '/home/hzy/Documents/dukto/CASIA-3D-FaceV1/3D-Face-WRL'
    dst_folder = '/home/hzy/Documents/work/preprocess/try'
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    dirs = os.listdir(wrl_folder)
    for dir in dirs:
        dirpath = os.path.join(wrl_folder, dir)
        files = os.listdir(dirpath)
        # dstfolder = os.path.join(dst_folder, dir)
        # if not os.path.exists(dstfolder):
        #     os.mkdir(dstfolder)
        for file in files:
            filepath = os.path.join(dirpath, file)
            x, y, z, alpha, beta, theta, r, g, b = wrl_reader(filepath)
            num = file.split(".")[0]
            point2gray(y, x, z, alpha, beta, theta, r, g, b, dst_folder, dir, num)
