# USAGE
# python LiveNet_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages

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

from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import torch
# construct the argument parse and parse the arguments
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from LiveNet import liveNet
from Res import modified_resnet
from updated_res import Resnet
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
"""
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())
"""
def get_rgb_adaptive_equalized(image):

    channels = cv2.split(image)
    eq_channels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(96,96))
    for ch, color in zip(channels, ['B','G','R']):
    	eq_channels.append(clahe.apply(ch))

    eq_image = cv2.merge(channels)
    return eq_image

print("Now Loading the LiveNet model for liveness detection.....");

model = Resnet()

model.load_state_dict(torch.load("./Resnet_Total_HE_with_Noise.pt"))

model.eval()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1500)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]
			face = get_rgb_adaptive_equalized(face)
			face = cv2.resize(face, (224, 224))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)


			# pass the face ROI through the trained liveness detector
			# additions by me
			# print(face.shape)

			# face = torch.from_numpy(face).float().to(device)

			# end of additions

			# print(face.shape);
			# Task to do here
			# Convert this face into a tensor 
			# change the dimension of face, just swap axis. Example [1,32,32,3] -> [1,3,32,32]
			# Load the pytorch model above instead of Keras Model
			# store the output of the model. 
			# determine the value of label according to the output.
			# model to determine if the face is "real" or "fake"

			face = torch.Tensor(face);
			face = face.permute(0,3,2,1)

			output = model(face)

			label = "real"

			if(output[0][0]>output[0][1]):
				label = "fake"


			# preds = model.predict(face)[0]
			#j = np.argmax(preds)
			#label = le.classes_[j]

			#print("Prediction Value = ",preds[j])
			#print("Label = ",label);

			# draw the label and bounding box on the frame
			# label = "{}: {:.4f}".format(label, preds[j])
			if(label=="real"):
				label = "{}: {:.4f}".format(label, output[0][1]*100)
			elif(label=="fake"):
				label = "{}: {:.4f}".format(label, output[0][0]*100)
			#label = "{}".format(label)
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()