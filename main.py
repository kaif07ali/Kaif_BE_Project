import os
import shutil
from datetime import datetime
import cv2
import pandas as pd
from matplotlib import pyplot as plt

input_dir="./data/Input/"
output_dir="./data/Output/"

# Create necessary folders if not already present
if not (os.path.isdir(input_dir)):
    os.mkdir(input_dir)
if not (os.path.isdir(output_dir)):
    os.mkdir(output_dir)

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# First clear contents output folders from previous run
folders = [output_dir]
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

outdir = input_dir + "input.jpeg"

start = datetime.now()
os.system('python detect.py --images ./data/Input/input.jpeg --dont_show')
end = datetime.now()

# Reading Pandas CSV File
yoloDf = pd.read_csv(output_dir + "ZipResults.csv")
base_out_path = output_dir
base_in_path = input_dir

# Iterating through the data
for index, row in yoloDf.iterrows():

    inp_fileName = row["filename"]
    out_filePath = base_out_path + row["filename"]
    inp_filePath = base_in_path + row["filename"]
    pred_prob = str(row["confidence"] * 100)[:4] + "%"
    count = row["count"]
    inp_label = inp_fileName
    is_defect_label = "Defective" if row["predlabel"] == 1 else "Ok"
    if is_defect_label == "Defective":
        pred_label = "PREDICTED LABEL: " + is_defect_label + f"\nScore: {pred_prob}" + f"\nError_Count: {count}"
    else:
        pred_label = "PREDICTED LABEL: " + is_defect_label
    inp_title = "Input Image"
    out_title = "Output Image"
    inp_image = cv2.imread(inp_filePath, cv2.IMREAD_UNCHANGED)
    out_image = cv2.imread(out_filePath, cv2.IMREAD_UNCHANGED)
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.title(inp_title)
    plt.imshow(cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB))
    plt.xlabel("Filename: " + inp_label, fontweight="bold", fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.title(out_title)
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.xlabel(pred_label, fontweight="bold", fontsize=15, color="green" if is_defect_label == "Ok" else "red")
    plt.xticks([])
    plt.yticks([])
    plt.show()

print("\n\nExecution Time: ", end - start)
