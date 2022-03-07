"""
This file is used for visualizing the frames from the data loader from keyframe
detection dataset
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--files_path',
    default='/Volumes/Storage/Temp/Ego4d_visualisation/fours/'
)
parser.add_argument(
    '--save_path',
    default='/Volumes/Storage/Temp/Ego4d_visualisation/fours/images/'
)
parser.add_argument(
    '--input_file',
    default=('/Volumes/Storage/Temp/Ego4d_visualisation/55b44f3a-bf40-'
             '48b1-be9c-a831ec5b0015.npy'),
    help='Path to file for single file mode'
)
parser.add_argument(
    '--single',
    default=1,
    help='If 1 code will run for a single file'
)
args = parser.parse_args()

if args.single == 1:
    data = np.load(open(args.input_file, 'rb'))
    save_folder = args.input_file.split('.')[0]
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    for count, image in enumerate(data):
        save_image = os.path.join(save_folder, str(count) + '.jpg')
        plt.imsave(save_image, image)
else:
    files_path = args.files_path
    save_path = args.save_path
    files = os.listdir(files_path)
    files = [os.path.join(files_path, item) for item in files]
    for file in files:
        data = np.load(open(file, 'rb'))
        images = list()
        save_folder = os.path.join(
            save_path, file.split('/')[-1].split('.')[0]
        )
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        for count, image in enumerate(data):
            save_image = os.path.join(save_folder, str(count) + '.jpg')
            plt.imsave(save_image, image)
