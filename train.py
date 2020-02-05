import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from model import UNet
from dataloader import DataLoader

def train_net(net,
              epochs=100,
              data_dir='data/',
              n_classes=3,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              gpu=True):
    
    ## path to save images
    output_dir = 'samples/'
    train_groundtruth = os.path.join(output_dir, 'train_groundtruth')
    train_input = os.path.join(output_dir, 'train_input')
    train_output = os.path.join(output_dir, 'train_output')
    
    test_groundtruth = os.path.join(output_dir, 'test_groundtruth')
    test_input = os.path.join(output_dir, 'test_input')
    test_output = os.path.join(output_dir, 'test_output')
    
    
    # load data
    loader = DataLoader(data_dir)

    N_train = loader.n_train()
    #N_train = 1
 
    ## change to adam 
    optimizer = optim.Adam(net.parameters(), 
                           #lr=lr,
                           #momentum=0.99,
                           weight_decay=0.0005)

    training_time = time.time()
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0
        
        ## load data
        (img, label) = enumerate(loader)
        
        image = torch.tensor(img)   # [160,4,128,128]
        label = torch.tensor(label) # [160,3,128,128]
        
        if gpu:
            image = image.cuda()
            label = label.cuda()
        
        batch = 16
        
        for i in range(0, int(N_train/batch)):
            ## get inputs x with batch 16 [16,4,128,128] [16,3,128,128]
            train_x = image[i*batch:(i+1)*batch,:,:,:]
            train_y = label[i*batch:(i+1)*batch,:,:,:]
            
            pred = net(train_x.float()) 
            
            optimizer.zero_grad()
            loss = optimizer(pred, train_y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            
            print('Training sample %d / %d - Loss: %.6f' % ((i+1)*batch, N_train, loss.item()))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))
        
    print('End of training. Time Taken: %d sec.' % (time.time() - training_time))
    
    Image.fromarray(label[-1,:,:,:].astype('uint8')).save(train_groundtruth)
    Image.fromarray(image[-1,:,:,:].astype('uint8')).save(train_input)
    Image.fromarray(pred[-1,:,:,:].astype('uint8')).save(train_output)
        
    
    # displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    with torch.no_grad():
        (img, label) = enumerate(loader)
        
        img_torch = torch.tensor(img) # [6,4,128,128]
        if gpu:
            img_test = img_torch.cuda()
        
        
        pred_test = net(img_test)  # [6,3,128,128]
        
        # plot test result
        for j in range(len(img)):
            self.plot(img[j,:,:,:], label[j,:,:,:], pred[j,:,:,:])
        
    Image.fromarray(label[-1,:,:,:].astype('uint8')).save(test_groundtruth)
    Image.fromarray(img_test[-1,:,:,:].astype('uint8')).save(test_input)
    Image.fromarray(pred_test[-1,:,:,:].astype('uint8')).save(test_output)

        
    def plot(self, img, label, pred):
        plt.subplot(1, 3, 1)
        plt.imshow(img*255.)
        plt.subplot(1, 3, 2)
        plt.imshow(label*255.)
        plt.subplot(1, 3, 3)
        plt.imshow(pred.cpu().detach().numpy().squeeze()*255.)
        plt.show()
        
        
        
        
        
        
        
        # for _, (img, label) in enumerate(loader):
        #     shape = img.shape
        #     img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
        #     if gpu:
        #         img_torch = img_torch.cuda()
        #     pred = net(img_torch)
        #     pred_sm = softmax(pred)
        #     _,pred_label = torch.max(pred_sm,1)

        #     plt.subplot(1, 3, 1)
        #     plt.imshow(img*255.)
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(label*255.)
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
        #     plt.show()

#def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    #print(p.size())
    return cross_entropy(p, target_label)

#def softmax(input):
    # todo: implement softmax function
    #input = input.resize_((2, input.size(2), input.size(3)))
    exp = torch.exp(input)
    sum_exp = torch.sum(exp,0)
    #sum_exp = sum_exp.resize_((1, 1, sum_exp.size(1), sum_exp.size(2)))
    p = exp/sum_exp
    
    return p

#def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    
    ## input [1,2,h,w]
    ## targets [h,w]
    #targets = torch.tensor(targets)
    #targets = targets.resize_((1, targets.size(0), targets.size(1)))
    
    M = input.size(1) * input.size(2)
    ce = -1 * torch.sum(targets * torch.log(input))/M

    return ce

# Workaround to use numpy.choose() with PyTorch
#def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[x,y], x, y]
            i += 1

    pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])

    return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=3, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir)
