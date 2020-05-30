import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import PIL
import os


TOTAL_BATCH_SIZE = 540
MINI_BATCH_SIZE = 10
LR = 0.005
show_every = False

dir_path = "D:/11PRojects/ML_DL/moire_dataset/data/"

dataset_x = np.zeros([TOTAL_BATCH_SIZE, 100, 100])
dataset_y = np.zeros([TOTAL_BATCH_SIZE, 100, 100])

for i in range(TOTAL_BATCH_SIZE):
    img_ori = cv2.imread(dir_path+str(i+1)+"original.jpg")
    img_ori = cv2.resize(img_ori, (100, 100), interpolation = cv2.INTER_CUBIC)
    img_moire = cv2.imread(dir_path+str(i+1)+"moire.jpg")
    img_moire = cv2.resize(img_moire, (100, 100), interpolation = cv2.INTER_CUBIC)
    # img_moire = img_moire / np.average(img_moire) * np.average(img_ori)

    #plt.imshow((img_ori[: ,: , 0] + img_ori[: ,: , 1] + img_ori[: ,: , 2])/3)
    #plt.show()
    #plt.imshow((img_moire[: ,: , 0] + img_moire[: ,: , 1] + img_moire[: ,: , 2])/3)
    #plt.show()

    dataset_y[i, :, :] = (img_ori[: ,: , 0] + img_ori[: ,: , 1] + img_ori[: ,: , 2])/3
    dataset_x[i, :, :] = (img_moire[: ,: , 0] + img_moire[: ,: , 1] + img_moire[: ,: , 2])/3

test_x = torch.unsqueeze(torch.tensor(dataset_x), dim=1).type(torch.cuda.FloatTensor)
test_y = torch.unsqueeze(torch.tensor(dataset_y), dim=1).type(torch.cuda.FloatTensor)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=32,    # n_filters
                kernel_size=11,      # filter size
                stride=1,           # filter movement/step
                padding=5,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(32, 1, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()   # the target label is not one-hotted

cnn.cuda()
loss_func.cuda()

# training and testing
for epoch in range(20):
    batch_start = int(np.random.rand()*(TOTAL_BATCH_SIZE-150))
    b_x = test_x[batch_start:batch_start+50]#[(step*20):(step*20+20)]
    b_y = test_y[batch_start:batch_start+50]#[(step*20):(step*20+20)]
    for step in range(30):
        # print(b_x, b_y)
        output = cnn(b_x)               # cnn output
        # print(output)
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        print(loss, output.size())

        if (step % 3 == 0 and show_every):
            #print(step*20+5)
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(b_y[5].cpu().data.numpy().squeeze(), cmap ='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(b_x[5].cpu().data.numpy().squeeze(), cmap ='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(output[5].cpu().data.numpy().squeeze(), cmap ='gray')
            plt.show()


test_output = cnn(test_x[490:500])
plt.figure()
for i in range(10):
    plt.subplot(4, 5, i*2+1)
    plt.imshow(test_output[i].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(4, 5, i*2+2)
    plt.imshow(test_y[490+i].cpu().data.numpy().squeeze(), cmap='gray')
plt.show()