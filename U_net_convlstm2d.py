#-*- coding:utf-8 -*-
"""
@Author   :Alex 
@Datetime :19-1-11 下午2:42
@contact: Bjc_alex.@163.com
@File name:segmentation-minify/U_net_convlstm2d
@Software : PyCharm
@Desc: CNN+ConvLSTM
@==============================@
@       ___   __    _  __      @
@      / _ | / /__ | |/_/      @
@     / __ |/ / -_)>  <        @
@    /_/ |_/_/\__/_/|_|        @
@                       常敦瑞  @
@==============================@
"""

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
from keras.layers.convolutional_recurrent import ConvLSTM2D


def get_unet(pretrained_weights=None, input_size=(None, 160, 240, 1)):
	inputs = Input(input_size)
	conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(inputs)
	conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1)
	pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
	conv2 = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool1)
	conv2 = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv2)
	pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
	conv3 = TimeDistributed(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool2)
	conv3 = TimeDistributed(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv3)
	# pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
	# conv4 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool3)
	# conv4 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4)
	drop4 = TimeDistributed(Dropout(0.5))(conv3)
	pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop4)
	
	conv5 = TimeDistributed(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool4)
	conv5 = TimeDistributed(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv5)
	drop5 = TimeDistributed(Dropout(0.5))(conv5)
	
	up6 = ConvLSTM2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
		TimeDistributed(UpSampling2D(size=(2, 2)))(drop5))
	merge6 = concatenate([drop4, up6], axis=4)
	# conv6 = ConvLSTM2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(merge6)
	# conv6 = ConvLSTM2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(conv6)
	
	# up7 = ConvLSTM2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
	# 	TimeDistributed(UpSampling2D(size=(2, 2)))(conv6))
	merge7 = concatenate([conv3, up6], axis=4)
	conv7 = ConvLSTM2D(256, 3, padding='same', return_sequences=True)(merge7)
	conv7 = ConvLSTM2D(256, 3, padding='same', return_sequences=True)(conv7)
	
	up8 = ConvLSTM2D(128, 2, padding='same',return_sequences=True)(
		TimeDistributed(UpSampling2D(size=(2, 2)))(conv7))
	merge8 = concatenate([conv2, up8], axis=4)
	conv8 = ConvLSTM2D(128, 3, padding='same', return_sequences=True)(merge8)
	conv8 = ConvLSTM2D(128, 3, padding='same', return_sequences=True)(conv8)
	
	up9 = ConvLSTM2D(64, 2, padding='same', return_sequences=True)(
		TimeDistributed(UpSampling2D(size=(2, 2)))(conv8))
	merge9 = concatenate([conv1, up9], axis=4)
	conv9 = ConvLSTM2D(64, 3, padding='same', return_sequences=True)(merge9)
	conv9 = ConvLSTM2D(64, 3, padding='same', return_sequences=True)(conv9)
	conv9 = TimeDistributed(Conv2D(2, 3, activation='relu', padding='same'))(conv9)
	# conv9 = ConvLSTM2D(2, 3, padding='same', return_sequences=True)(conv9)
	# conv10 = ConvLSTM2D(3, 1, activation='softmax', return_sequences=True)(conv9)
	conv10 = TimeDistributed(Conv2D(2, 1,activation='softmax', padding='same'))(conv9)
	
	model = Model(input=inputs, output=conv10)
	
	model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
	
	# plot_model(model, to_file='MRI_brain_seg_UNet3D.png', show_shapes=True)
	# model.summary()
	
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	
	return model
