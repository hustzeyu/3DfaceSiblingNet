#coding:UTF-8
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ply_reader(ply_file):
    with open(ply_file, "r") as f:
        lines = f.readlines()
        lines = lines[12:-3]
    x = []
    y = []
    z = []
    for line in lines:
        parts = line.split(" ")
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        z.append(float(parts[2]))
    return x, y, z

def obj_reader(obj_file):
    alpha, beta, theta, x, y, z = [], [], [], [], [], []
    with open(obj_file, "r") as f:
        lines = f.readlines()
        lines = [i for i in lines if i[0]=="v"]
    for line in lines:
        if line[:2] == "vn":
            parts = line.rstrip("\n").split(" ")
            alpha.append(float(parts[1]))
            beta.append(float(parts[2]))
            theta.append(float(parts[3]))
        else:
            parts = line.rstrip("\n").split(" ")
            x.append(float(parts[1]))
            y.append(float(parts[2]))
            z.append(float(parts[3]))
    return x, y, z, alpha, beta, theta

def points_show(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X label',color='r')
    ax.set_ylabel('Y label',color='r')
    ax.set_zlabel('Z label')
    ax.scatter(x,y,z,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')
    plt.show()


if __name__ == "__main__":
    x, y, z, alpha, beta, theta = obj_reader("30.obj")
    print x, y, z, alpha, beta, theta
