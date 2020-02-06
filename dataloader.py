import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image
import torch

from torchvision import transforms
from matplotlib import cm

import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2):
        self.batch_size = batch_size
        #self.test_percent = test_percent

        self.data_dir = abspath(root_dir)
        #self.data_dir = join(self.root_dir, 'scans')
        #self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        #self.label_files = [join(self.labels_dir, f) for f in self.files]
        

    def __iter__(self):
        #n_train = self.n_train()

        if self.mode == 'train':
            current = 1
            crop_num = 80
            #endId = n_train
            
        elif self.mode == 'test':
            current = 0
            crop_num = 3
            #endId = len(self.data_files)
            
        image = Image.open(self.data_files[current])
        #label_image = Image.open(self.label_files[current-1])
        
        #### Generate data ####
        width, length = image.size # [430, 870]
        
        image = np.array(image)  # [870,430,3]
        
        inputs = np.zeros((crop_num*2, 4, 128, 128))
        outputs = np.zeros((crop_num*2, 3, 128, 128))
        
        for i in range(0, crop_num):
            # crop small patch
            crop_l = 128
            crop_w = 128
            
            x = random.randint(0, length-crop_l)
            y = random.randint(0, width-crop_w)
            
            data_image = image[x:x+crop_l, y:y+crop_w, :] # [128,128,3]
            
            # normalize to [0,1]
            output1 = data_image / np.max(data_image)  # [128,128,3]
            
            # convert to [3,128,128]
            output1 = np.resize(output1,(1,3,crop_l,crop_w))
            
            # random augmentation
            
            data_image = Image.fromarray(np.uint8(data_image))
            
            # horizontal flip
            transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.ColorJitter(brightness = 2),
                                            transforms.RandomRotation(90)]) 
            
            output2 = np.array(transform(data_image))
            # normalize to [0,1]
            output2 = output2 / np.max(output2)  # [3,128,128]
            
            # convert to [3,128,128]
            output2 = np.resize(output2,(1,3,crop_l,crop_w))
            
            # add random holes on image
            
            input1 = self.addrandomhole(output1)
            input2 = self.addrandomhole(output2)
            
            inputs[i,:,:,:] = input1
            inputs[i+crop_num,:,:,:] = input2
            
            outputs[i,:,:,:] = output1
            outputs[i+crop_num,:,:,:] = output2
            
        yield (inputs, outputs)  # [160,4,128,128] [160,3,128,128] 
    
    def addrandomhole(self, image): 
        # image [1,3,128,128] RGB
        # result [1,4,128,128] RGB with mask
        i = 0
        size = 128
        
        image = np.resize(image,(3,size,size))
        
        while i < 5: # add 5 holes in each image
            # generate mask
            mask = np.ones((size, size))
            
            long = random.randint(0, size-64)
            short = random.randint(0, size-8)
            
            j = random.randint(0,1)
            if j == 0: #[64,8]
                hole = np.zeros((64,8))
                mask[long:long+64, short:short+8] = hole
                
            elif j ==1: #[8,64]
                hole = np.zeros((8,64))
                mask[short:short+8, long:long+64] = hole
            
            # add mask on image
            image[:,mask==0] = 0 
            
            #resize mask
            mask = np.resize(mask, (1, size, size))
            
            image = np.vstack([image, mask])
            
            i+=1
        
        image = np.resize(image,(1,4,size,size))
        
        return image

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.outputs)
        return data_length