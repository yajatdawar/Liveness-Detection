# USAGE
# python train_resnet.py --dataset dataset --model livenet.model --le le.pickle

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
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
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-e", "--epochs", type=str, required=True,
	help="number of epochs to train the model")
ap.add_argument("-s", "--saved_model", type=str, required=True,
	help="name of the model to be saved")
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
print("loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_dataset():
    data_path = 'dataset'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def get_rgb_adaptive_equalized(image):

    channels = cv2.split(image)
    eq_channels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(96,96))
    for ch, color in zip(channels, ['B','G','R']):
    	eq_channels.append(clahe.apply(ch))

    eq_image = cv2.merge(channels)
    return eq_image

def get_rgb_equalized(image):
	channels = cv2.split(image)
	eq_channels = []
	for ch, color in zip(channels, ['B', 'G', 'R']):
		eq_channels.append(cv2.equalizeHist(ch))
	eq_image = cv2.merge(eq_channels)
	return eq_image	

def add_gaussian_noise(image):
	row,col,ch= image.shape
	mean = 0
	var = 0.1
	sigma = var**0.5
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy = image + gauss
	return noisy

for imagePath in imagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 224x224 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	
	image = get_rgb_adaptive_equalized(image)
	image = add_gaussian_noise(image)
	image = cv2.resize(image, (224, 224))

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tensor_x = torch.stack([torch.Tensor(i) for i in data]) # transform to torch tensors
tensor_y = torch.LongTensor(labels)
# tensor_y = torch.stack([torch.LongTensor(i) for i in labels])

train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset


num_train = len(train_dataset)

indices = list(range(num_train))

#np.random.shuffle(indices)

valid_size = 0.25

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_dataloader = utils.DataLoader(train_dataset,batch_size = BS,sampler = train_sampler,num_workers = 0) 

valid_loader = utils.DataLoader(train_dataset, batch_size=BS,sampler = valid_sampler, num_workers= 0)

#model = modified_resnet()
#model = modified_inception()
model = Resnet()
model.load_state_dict(torch.load(args["model"]))
model.train()
import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = args["epochs"]
num_epochs = int(n_epochs,10)
print("Number of Epochs = ",n_epochs)
Total_Accuracy = 0
Total_Val_Accuracy = 0

max_total_accuracy = 0
max_val_accuracy = 0

max_final_accuracy = 0


for epoch in range(1,num_epochs+1):

	train_loss = 0.0

	valid_loss = 0.0

	correct_train = 0
	total_train = 0

	correct_val = 0
	total_val = 0

	model.train()

	for data,target in train_dataloader:

		optimizer.zero_grad()

		data = data.permute(0,3,2,1)

		#print("Input is ",data)

		#print("Actual Labels = ",target)

		output = model(data)

		temp = torch.max(output,1)

		prediction = temp.indices

		#print("Prediction = ",temp.indices)
		#print("Actual = ",target)

		for i in range(len(target)):

			if(target[i]==prediction[i]):
				correct_train = correct_train+1

			total_train	 = total_train+1

		#print("Total Correct = ",correct_train)
		#print("output = ",output)
		loss = criterion(output,target)

		loss.backward()

		optimizer.step()

		train_loss += loss.item()*data.size(0)

	train_loss = train_loss/len(train_dataloader.sampler)

	Training_accuracy = (correct_train/total_train)*100.0

	Total_Accuracy = Total_Accuracy + Training_accuracy

	model.eval()

	for data,target in valid_loader:

		data = data.permute(0,3,2,1)

		output = model(data)
		prediction = (torch.max(output,1)).indices

		loss = criterion(output,target)

		for i in range(len(target)):
			if(target[i]==prediction[i]):
				correct_val = correct_val+1

			total_val = total_val+1

		valid_loss += loss.item()*data.size(0)

	valid_loss = (valid_loss/len(valid_loader.sampler))

	Val_Accuracy = (correct_val/total_val)*100.0									

	Total_Val_Accuracy = Total_Val_Accuracy + Val_Accuracy
	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\t Training Accuracy:{:.6f}\t Validation Accuracy {:.6f} '.format(epoch, train_loss, valid_loss,Training_accuracy,Val_Accuracy))

	if(Val_Accuracy+Training_accuracy>= max_final_accuracy):
		print("Model saved at epoch ",epoch)
		torch.save(model.state_dict(),args["saved_model"])
		max_final_accuracy = Val_Accuracy+Training_accuracy



   

print("Average Training accuracy = ",Total_Accuracy/num_epochs)
print("Average Validation Accuracy = ",Total_Val_Accuracy/num_epochs)




# 1 is for real
# 0 is for fake


# Evaluation script
"""
total_correct_values = 0
total_values = 0
i=0
model = modified_resnet()

while i<data_len:

	#tensor_x[i] = tensor_x[i].unsqueeze(0)

	# print(tensor_x[i].unsqueeze(0).shape)
	model.eval()
	predicted_label = 0
	output = model(tensor_x[i].unsqueeze(0).permute(0,3,2,1))
	print(output.shape)
	i+= 1
	if(i>50):
		break

	if(output[0][1]>output[0][0]):
		predicted_label = 1

	if(predicted_label==labels[i]):
		total_correct_values+= 1

	total_values+= 1

	print("index = ",i,"Predicted Label = ",predicted_label)
	print("index = ",i,"Actual label = ",labels[i])
	i+= 50

# print((total_correct_values/total_values)*100.0)

"""
# End of Evaluation Script






