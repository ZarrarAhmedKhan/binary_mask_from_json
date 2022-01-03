import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--json_file', help='json annotations')
parser.add_argument('--images_folder', help = 'Folder containing images')
parser.add_argument('--output_folder', help = 'Output folder containing result images')
args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

with open(args.json_file, "r") as read_file:
    data = json.load(read_file)

all_file_names=list(data.keys())

Files_in_directory = []
for root, dirs, files in os.walk(args.images_folder):
    for filename in files:
        Files_in_directory.append(filename)
        
for j in range(len(all_file_names)): 
    image_name=data[all_file_names[j]]['filename']
    if image_name in Files_in_directory: 
         img = np.asarray(PIL.Image.open('sample_frames/'+image_name))    
    else:
        continue
    
    if data[all_file_names[j]]['regions'] != {}:
        #cv2.imwrite('images/%05.0f' % j +'.jpg',img)
        img_name = image_name.split('.')[0]
        print(img_name)
        try: 
             shape1_x=data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_x']
             shape1_y=data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_y']
        except : 
             shape1_x=data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_x']
             shape1_y=data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_y']
    
        fig = plt.figure()
      
        plt.imshow(img.astype(np.uint8)) 
        plt.scatter(shape1_x,shape1_y,zorder=2,color='red',marker = '.', s= 55)
        

        ab=np.stack((shape1_x, shape1_y), axis=1)
        
        img2=cv2.drawContours(img, [ab], -1, (255,255,255), -1)
       
        
        
        mask = np.zeros((img.shape[0],img.shape[1]))
        img3=cv2.drawContours(mask, [ab], -1, 255, -1)
        
        cv2.imwrite(f'{args.output_folder}/{img_name}.png',mask.astype(np.uint8))
        
    