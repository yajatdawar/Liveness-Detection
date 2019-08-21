import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class liveNet(nn.Module):

	def __init__(self):

		super(liveNet,self).__init__()

		# first CONV => RELU => CONV => RELU => POOL layer set

		# Initialy the first layer sees 32x32x3 image tensor
		self.conv11 = nn.Conv2d(3,16,3,stride = 1,padding = 1)
		self.conv11_bn = nn.BatchNorm2d(16)
		# 32x32x16 image tensor
		self.conv12 = nn.Conv2d(16,16,3,stride = 1,padding = 1)
		self.conv12_bn = nn.BatchNorm2d(16)
		# 32x32x16 image tensor
		self.pool1 = nn.MaxPool2d(2,2)

		# Final Output from 1st Layer
		# 16x16x16 image tensor

		# second CONV => RELU => CONV => RELU => Pooling Layer 

		# 16x16x16 image tensor
		self.conv21 = nn.Conv2d(16,32,3,stride = 1,padding = 1)
		self.conv21_bn = nn.BatchNorm2d(32)
		# 16x16x32 image tensor
		self.conv22 = nn.Conv2d(32,32,3,stride = 1,padding = 1)
		self.conv22_bn = nn.BatchNorm2d(32)
		# 16x16x32 image tensor
		self.pool2 = nn.MaxPool2d(2,2)

		# Final Output from 2nd Layer
		# 8x8x32 image tensor


		# Finally we have two fully connected Layer
		self.fc1 = nn.Linear(8*8*32,100)

		self.fc2 = nn.Linear(100,10)

		self.fc3 = nn.Linear(10,2)

		self.dropout = nn.Dropout(0.25)

		self.soft = nn.Softmax()

		self.bn1 = nn.BatchNorm1d(num_features = 100)

		self.bn2 = nn.BatchNorm1d(num_features = 10)

	def forward(self,x):

		x = self.pool1(F.relu(self.conv12_bn(self.conv12(F.relu(self.conv11_bn(self.conv11(x)))))))

		x = self.dropout(x)

		x = self.pool2(F.relu(self.conv22_bn(self.conv22(F.relu(self.conv21_bn(self.conv21(x)))))))

		x = self.dropout(x)


		x = x.view(-1,8*8*32)

		x = F.relu(self.fc1(x))

		x = F.relu(self.fc2(x))

		x = self.dropout(x)

		x = self.fc3(x)

		x = self.soft(x)

		return x







