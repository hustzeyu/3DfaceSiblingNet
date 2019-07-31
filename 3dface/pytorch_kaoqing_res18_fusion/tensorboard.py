#coding:UTF-8
from tensorboardX import SummaryWriter
import torch
import torchvision
import cv2
import numpy as np

class TensorBoard(object):
    """
    visualize the acc,loss,model,imgs
    """
    def __init__(self, batch, channel, height, width):
        self.batch = batch
        self.channel = channel
        self.height = height
        self.width = width
        self.writer = SummaryWriter()

    def visual_model(self, model):
        dummy_input = torch.rand(self.batch, self.channel, self.height, self.width)
        with SummaryWriter(comment="model") as w:
            w.add_graph(model, (dummy_input,))

    def visual_loss(self, loss_name, loss, epoch):
        self.writer.add_scalar(loss_name, loss, epoch)

    def visual_acc(self, acc_name, acc, epoch):
        self.writer.add_scalar(acc_name, acc, epoch)

    def visual_img(self, batch_data, epoch):
        img = torchvision.utils.make_grid(batch_data)
        self.writer.add_image("image", img, epoch)
        
        
       

        
