"""
Downloads the moon and mars models locally if they don't exist.

Returns:
    None
"""

# Python File to Download Models Locally

import os
import subprocess

directory = "models"
path = os.path.join(os.getcwd(), directory)  # Gets current directory

if not os.path.exists(path):  # Checks if there already is Models Folder
    os.makedirs(path)

subprocess.run(["pip", "install", "gdown"])  # Install gdown

left_model_path = os.path.join(path, "Moon_Left_Model.pt")
if not os.path.exists(left_model_path):
    print('Donwloading First Model for Moon')
    subprocess.run([
        "gdown",
        "13C6Spdgb5Vrx7uLkDE3hAk7E2qChfdAd",
        "-O",
        left_model_path
    ])  # Downloads left side of moon model
    print('Done')
    print()

right_model_path = os.path.join(path, "Moon_Right_Model.pt")
if not os.path.exists(right_model_path):
    print('Donwloading Second Model for Moon')
    subprocess.run([
        "gdown",
        "16-vgOA_YDARJGGC6xHo9f1GpL8SFGch-",
        "-O",
        right_model_path
    ])  # Downloads right side moon model
    print('Done')
    print()

mars_model_path = os.path.join(path, "Mars_best.pt")
if not os.path.exists(mars_model_path):
    print('Donwloading Best Model for Mars')
    subprocess.run([
        "gdown",
        "1-q8bZw-gjgWrb2yaZgFbngiDZtslb28B",
        "-O",
        mars_model_path
    ])  # Downloads best mars model (Large Model)
    print('Done')

print('All Models have been downloaded')
