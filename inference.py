#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[1]:


import os, shutil

from IPython import get_ipython
from IPython.display import Image, display
from matplotlib import pyplot as plt
from ipywidgets import FileUpload
import zipfile
import matplotlib.image as mpimg
import math
from matplotlib.widgets import Button
import numpy as np
import cv2
from datetime import datetime
import pandas as pd

input_dir="./data/Input/"
output_dir="./data/Output/"

#Create necessary folders if not already present
if not(os.path.isdir(input_dir)):
    os.mkdir(input_dir)
if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)

#Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# # Delete previous inputs and upload user input

# In[2]:


#First clear contents of input and output folders from previous run
folders = [input_dir,output_dir]
for folder in folders:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))  

# In[3]:

print("\n\nWelcome to DiscBrake Fault Detector")
print("\n\n\tMENU\n")
print("1. Single Image")
print("2. Batch Processing .ZIP file")
choice=int(input("Enter your choice (1-2)?"))
if(choice==1):
    outdir=input_dir + "input.jpeg"
elif(choice==2):
    outdir=input_dir + "input.zip"
print("Saved at ",outdir)

def on_upload_change(change):
    if not change.new:
        return
    up = change.owner
    for filename,data in up.value.items():
        print('Upload Successful!')
        with open(outdir, 'wb') as f:
           f.write(data['content'])
    up.value.clear()
    up._counter = 0

upload_btn = FileUpload()
upload_btn.observe(on_upload_change, names='_counter')
upload_btn


# # Calling Trained Model

# In[6]:


#image processing and labelling
start=datetime.now()
if(choice==1):
    #For Single Image
    get_ipython().system('python detect.py --images ./data/Input/input.jpeg --dont_show')

elif(choice==2):
    #For ZIP File
    with zipfile.ZipFile(outdir, 'r') as zip_ref:
        zip_ref.extractall(input_dir)
    os.remove(input_dir+"input.zip")
    get_ipython().system('python detect.py --images ./data/Input/ --zip --dont_show')
end=datetime.now()


# # Model Output

# In[7]:


# Reading Pandas CSV File
yoloDf = pd.read_csv(output_dir+"ZipResults.csv")
base_out_path = output_dir
if(choice==1):
    base_in_path = input_dir
else:
    base_in_path = input_dir

# Iterating through the data
for index, row in yoloDf.iterrows():

    inp_fileName = row["filename"]
    out_filePath = base_out_path + row["filename"]
    inp_filePath = base_in_path + row["filename"]
    pred_prob = str(row["confidence"]*100)[:4]+ "%"
    count = row["count"]
    inp_label = inp_fileName
    is_defect_label = "Defective" if row["predlabel"] == 1 else "Ok"
    if(is_defect_label=="Defective"):
        pred_label = "PREDICTED LABEL: " + is_defect_label  + f"\nScore: {pred_prob}" + f"\nError_Count: {count}"
    else:
        pred_label = "PREDICTED LABEL: " + is_defect_label
    inp_title = "Input Image"
    out_title = "Output Image"
    inp_image = cv2.imread(inp_filePath, cv2.IMREAD_UNCHANGED)
    out_image = cv2.imread(out_filePath, cv2.IMREAD_UNCHANGED)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.title(inp_title)
    plt.imshow(cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB))
    plt.xlabel("Filename: "+inp_label,fontweight = "bold", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.title(out_title)
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.xlabel(pred_label,fontweight = "bold", fontsize = 15,color = "green" if is_defect_label == "Ok" else "red")
    plt.xticks([])
    plt.yticks([])
    plt.show()


print("\n\nExecution Time: ",end-start)


# In[ ]:




