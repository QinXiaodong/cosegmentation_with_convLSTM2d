#coding=utf-8
# paoso@qq.com
# 2019-5-26
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2 as cv

origin_data_path='origin_data'
processed_data_path='data'
target_size=(64,64)
sample_dirs=os.listdir(os.path.join(origin_data_path,'images'))
for sample_dir in sample_dirs:
    # 读取一个样本中所有文件
    image_files=os.listdir(os.path.join(origin_data_path,'images',sample_dir))
    # 保存一个样本中所有文件名，不包含后缀
    image_names=[]
    for img in image_files:
        image_names.append(os.path.splitext(img)[0])

    sample_input=np.zeros((len(image_names),)+target_size+(3,)) #(None,360,360,3) 存储图片序列
    sample_target=np.zeros((len(image_names),)+target_size+(2,)) #(None,360,360.2) 存储目标序列

    sample_input=np.zeros((3,)+target_size+(3,)) #(None,360,360,3) 存储图片序列
    sample_target=np.zeros((3,)+target_size+(2,)) #(None,360,360.2) 存储目标序列
    
    
    img_index=0
    for image_name in image_names:
        if img_index>2:
            break
        # 读取原始图像文件和groundtruth文件
        temp_img=load_img(os.path.join(origin_data_path,'images',sample_dir,image_name+'.jpg'),target_size=target_size)
        temp_target=load_img(os.path.join(origin_data_path,'targets',sample_dir,image_name+'.png'),target_size=target_size,color_mode='grayscale')
        # 转成numpy array
        temp_img=img_to_array(temp_img)
        temp_target=img_to_array(temp_target)

        #存储
        sample_input[img_index,:,:,:]=temp_img
        sample_target[img_index,:,:,0][temp_target[:,:,0]>0]=1 # 0通道 0代表背景，1代表前景
        sample_target[img_index,:,:,1][temp_target[:,:,0]==0]=1 # 1通道 0代表前景，1代表背景
        
        img_index+=1
    np.savez(os.path.join(processed_data_path,sample_dir),input=sample_input,target=sample_target)

    
