# USAGE
# python test.py --test_dataset test_dataset --model torch_model.pt --le le.pickle

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as utils
import matplotlib		
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

from LiveNet import liveNet
from Res import modified_resnet
from updated_res import Resnet
import torchvision.models as models
resnet18 = models.resnet18()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test_dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("loading images from Test dataset...")
imagePaths = list(paths.list_images(args["test_dataset"]))
data = []
labels = []

def get_rgb_equalized(image):
	channels = cv2.split(image)
	eq_channels = []
	for ch, color in zip(channels, ['R', 'G', 'B']):
		eq_channels.append(cv2.equalizeHist(ch))
	eq_image = cv2.merge(eq_channels)
	return eq_image	

def get_rgb_adaptive_equalized(image):

    channels = cv2.split(image)
    eq_channels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(96,96))
    for ch, color in zip(channels, ['B','G','R']):
    	eq_channels.append(clahe.apply(ch))

    eq_image = cv2.merge(channels)
    return eq_image

for imagePath in imagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = get_rgb_adaptive_equalized(image)
	image = cv2.resize(image, (224,224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data_len = len(data)
print(data_len)
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
num_labels = len(labels)

# labels = labels.reshape(num_labels,1)

#labels = np_utils.to_categorical(labels, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tensor_x = torch.stack([torch.Tensor(i) for i in data]) # transform to torch tensors
tensor_y = torch.LongTensor(labels)
# tensor_y = torch.stack([torch.LongTensor(i) for i in labels])

train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset

# 1 is for real
# 0 is for fake

# Evaluation script

total_correct_values = 0
total_values = 0
i=0
model = Resnet()

model.load_state_dict(torch.load(args["model"]))

model.eval()

while i<data_len:

	#tensor_x[i] = tensor_x[i].unsqueeze(0)

	# print(tensor_x[i].unsqueeze(0).shape)
	predicted_label = 0
	output = model(tensor_x[i].unsqueeze(0).permute(0,3,2,1))

	if(output[0][1]>output[0][0]):
		predicted_label = 1

	if(predicted_label==labels[i]):
		total_correct_values+= 1

	total_values+= 1

	i+= 1

print("Accuracy = ",(total_correct_values/total_values)*100)

# End of Evaluation Script