# Python Script For Automation of the tasks

# Step1 
"""
Assumptions
A folder named "videos" is present
it has two folders "Real_Videos" and "Fake_Videos"

In this step the task is to extract real faces and fake faces from these videos and put these images into the folder "dataset",
divided into two folders "real" and "fake". Also a test_dataset is generated.
"""

# Step2
"""
In this step the generated dataset is used to train an already trained pytorch model and save the model with the name 
provided. 
"""

# Step3
"""
This step takes the trained model and test it on the test dataset generated in step1 and publish the accuracy of
training and testing.
"""

# Step4
"""
delete the generated dataset so that when new data comes, the old one is removed.
"""
import json
import os


def pause():
    programPause = input("Press the <ENTER> key to continue...")



# read file
with open('arguments.json', 'r') as myfile:
    data=myfile.read()

# parse file
argument = json.loads(data)

# For creating the dataset from videos
str1 = "python gather_examples.py"

str1+= " --real "
str1+= argument["real"]

str1+= " --fake "
str1+= argument["fake"]

str1+= " --outreal "
str1+= argument["outreal"]

str1+= " --outfake "
str1+= argument["outfake"]

str1+= " --detector "
str1+= argument["detector"]

str1+= " --skip "
str1+= argument["skip"]

os.system(str1)
#Example
#os.system("python gather_examples.py --real videos/sample_real --fake videos/sample_fake --outreal videos/real_pics --outfake videos/fake_pics --detector face_detector --skip 4")

# End of Step 1

print("Training Dataset is saved. Please check the dataset before going forward !! . Remove any invalid data !!\n")

pause()

print("Generating Test Dataset\n")
# Step2

str4 = "python gather_examples.py"

str4+= " --real "
str4+= argument["real_test"]

str4+= " --fake "
str4+= argument["fake_test"]

str4+= " --outreal "
str4+= argument["outreal_test"]

str4+= " --outfake "
str4+= argument["outfake_test"]

str4+= " --detector "
str4+= argument["detector"]

str4+= " --skip "
str4+= argument["skip"]
# End of Step2
os.system(str4)

print("Test Dataset is generated\n")
print("Test Dataset is saved. Please check the dataset before going forward !! . Remove any invalid data !!\n")

pause()
print("\nTraining is getting started... \n")
# python train_resnet.py --dataset dataset --model ./torch_model.pt --epochs 10 --saved_model torch_new.pt --le le.pickle

# Step3

str2 = "python train_resnet.py"

str2+= " --dataset "
str2+= argument["dataset"]

str2+= " --model "
str2+= argument["model"]

str2+= " --epochs "
str2+= argument["epochs"]

str2+= " --saved_model "
str2+= argument["saved_model"]

str2+= " --le "
str2+= argument["le"]

os.system(str2)

print("End of Training\n")

print("The trained model is saved as ",argument["saved_model"])

# End of Step 3

# Step 4
# In this step trained model is tested on a test_dataset to print the accuracy of the model
# python test.py --test_dataset test_dataset --model torch_model.pt --le le.pickle

print("\nEvaluation is starting....\n")

str3 = "python test.py"

str3+= " --test_dataset "
str3+= argument["test_dataset"]

str3+= " --model "
str3+= argument["saved_model"]

str3+= " --le "
str3+= argument["le"]

os.system(str3)

print("End of Evaluation\n")
# End of Step4



