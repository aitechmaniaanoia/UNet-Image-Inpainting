import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.down1 = downStep(4, 64) 
        self.down2 = downStep(64, 128) 
        self.down3 = downStep(128, 256) 
        self.down4 = downStep(256, 512) 
        
        self.conv1 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1)
        
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64)
        
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size = 1)

    def forward(self, x):
        # todo
        d1,d1_down = self.down1(x)
        d2, d2_down = self.down2(d1)
        d3, d3_down = self.down3(d2)
        d4, d4_down = self.down4(d3)
        
        f = F.relu(self.conv1(d4))
        f = self.conv2(f)
        
        u1 = self.up1(f, d4_down)
        u2 = self.up2(u1, d3_down)
        u3 = self.up3(u2, d2_down)
        u4 = self.up4(u3, d1_down)
        
        x = F.sigmoid(self.conv3(u4))
        
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.conv1 = nn.Conv2d(inC, outC, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size = 3, padding = 1)
        
        self.maxpool = nn.MaxPool2d( kernel_size = 2, stride = 2)
        
    def forward(self, x):
        # todo
        x = F.relu(self.conv1(x))

        x_down = F.relu(self.conv2(x))

        x = self.maxpool(x_down)
        
        return x, x_down

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.up = nn.ConvTranspose2d(inC, outC, kernel_size = 2, stride = 2) # stride = 2
        
        self.conv1 = nn.Conv2d(inC, outC, kernel_size = 3, padding = 1) 
        self.conv2 = nn.Conv2d(outC, outC, kernel_size = 3, padding = 1)
        
        #self.batch = nn.BatchNorm2d(outC)

    def forward(self, x, x_down):
        # todo
        x = self.up(x)

        #print(x.size())
        #print(x_down.size())
        
        x = torch.cat((x_down, x), dim = 1)
        #print(x.size())
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        
        return x
    
    
    