import open3d as o3d
import os
import numpy as np
import cv2
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

flag = 0
def get_normals(pcd):
    o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30, max_nn=10))
    return pcd

def main(path):
    xyz = []
    x1, y1, z1, r1, g1, b1 = obj_reader(path)
    if len(x1) == 0:
        print ('len(x1) = 0')
        return []
    loop = len(x1)
    print ("len1 = ", loop)
    for i in range(loop):
        point1 = [x1[i], y1[i], z1[i]]
        xyz.append(point1)
    axyz = np.array(xyz)
    bxyz = axyz[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bxyz)
    o3d.visualization.draw_geometries([pcd])
    down_pcd = get_normals(pcd)
    normals = down_pcd.normals
    npnormals = np.array(normals)
    np.save("/home/hzy/Documents/data/try/hah.npy", npnormals)
    path2 = "/home/hzy/Documents/data/try/downpcd.txt"
    with open(path2, "w") as f:
        f.write(str(npnormals))
    print ("npnormals = ", npnormals)
    return npnormals

def obj_reader(obj_file):
    x, y, z, r, g, b = [], [], [], [], [], []
    # alpha, beta, theta = [], [], []
    with open(obj_file, "r") as f:
        lines = f.readlines()
        lines = [ele for ele in lines if ele[0] == "v"]
        lines = [''.join(ele) for ele in lines]
        lines = [ele.rstrip("\n") for ele in lines]
        lines = [ele.split(' ') for ele in lines]
        for line in lines:
            x.append(float(line[1]))
            y.append(float(line[2]))
            z.append(float(line[3]))
            r.append(float(line[4]))
            g.append(float(line[5]))
            b.append(float(line[6]))
    return x, y, z, r, g, b

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
    path = '/home/hzy/Documents/data/try/003.obj'
    # path = '/home/hzy/Documents/data/try/30.obj'
    out = main(path)
    print ('ok')
