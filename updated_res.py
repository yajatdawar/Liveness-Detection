import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Resnet(nn.Module):

	def __init__(self):

		super(Resnet,self).__init__()

		# Initialy the first layer sees 32x32x3 image tensor
		self.res = models.resnet18(num_classes=2)
		self.soft = nn.Softmax()
		
	def forward(self,x):

		x = self.res(x)
		x = self.soft(x)
		return x




"""
import torch
import torch.nn as nn
import torchvision.models as models

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.resnet = models.resnet50(num_classes=128)
    
    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.normalize(x, dim=1)
        return x


"""