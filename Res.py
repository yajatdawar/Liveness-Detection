import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
resnet18 = models.resnet18()

class modified_resnet(nn.Module):

	def __init__(self):

		super(modified_resnet,self).__init__()

		# Initialy the first layer sees 32x32x3 image tensor
		self.res = resnet18
		self.fc = nn.Linear(1000,2)

		
	def forward(self,x):

		x = self.fc(self.res(x))
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