import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
image = plt.imread('input.jpg')
in_image = image/255

## Convolution Function
def Conv_1(Input, Weight, Bias): #Input(row, column, channel)
    ch = len(Weight)
    ch_before = len(Weight[0])
    conv_out = np.zeros((ch, len(Input), len(Input)))
    Input_pad = np.pad(Input, ((1,1),(1,1),(0,0)),mode='constant', constant_values = (0,0))
    
    for i in range(0, ch): #channel
        print("Now we are at channel:", i, "Total:", ch)
        for j in range(0, len(Input)): #ROW+j
            for k in range(0, len(Input[0])): #COL+k
                conv_out[i][j][k] = conv_out[i][j][k] + Bias[i]
                for l in range(0, ch_before): # num of Channel Before
                    for m in range(0, len(Weight[0][0])): #ROW
                        for n in range(0, len(Weight[0][0][0])): #COL
                            conv_out[i][j][k] = Input_pad[m+j][n+k][l] * Weight[i][l][m][n] + conv_out[i][j][k]
    return conv_out

def Conv(Input, Weight, Bias): #Input(channel, row, column), Weight(ch after, ch before, r, c), Bias(ch after)
    ch = len(Weight)
    ch_before = len(Input)
    conv_out = np.zeros((ch, len(Input[0]), len(Input[0][0]))) 
    Input_pad = np.pad(Input, ((0,0),(1,1),(1,1)),mode='constant', constant_values = (0,0))
    for i in range(0, ch): #channel
        print("Now we are at channel:", i, "Total:", ch)
        for j in range(0, len(Input[0])): #ROW+j
            for k in range(0, len(Input[0][0])): #COL+k
                conv_out[i][j][k] = conv_out[i][j][k] + Bias[i] 
                for l in range(0, ch_before): #ch before
                    for m in range(0, len(Weight[0][0])): #row
                        for n in range(0, len(Weight[0][0][0])): #col
                            conv_out[i][j][k] = Input_pad[l][m+j][n+k] * Weight[i][l][m][n] + conv_out[i][j][k]
    return conv_out
  
## ReLU activation function
def ReLU(Input):
    ch = len(Input)
    r = len(Input[0])
    c = len(Input[0][0])
    Result = np.zeros((ch, r, c))
    for i in range(0, ch): #CHANNEL
        for j in range(0, r): #ROW
            for k in range(0, c): #COL
                if Input[i][j][k] < 0:
                    Result[i][j][k] = 0
                else:
                    Result[i][j][k] = Input[i][j][k]
    return(Result)

def ReLU_FC(Input):
    ch = len(Input)
    Result = np.zeros((ch))
    for i in range(0, ch):
        if Input[i] < 0:
            Result[i] = 0
        else:
            Result[i] = Input[i]
    return Result
  
#MAXPOOL FUNCTION
def Maxpool2d(Input):
    temp_max = 0
    ch = len(Input)
    r = len(Input[0])
    c = len(Input[0][0])
    maxpool_out = np.zeros((ch, int(r/2), int(c/2)))
    for i in range(0, ch): #CHANNEL
        for j in range(0, r, 2): #ROW stride=2
            for k in range(0, c, 2): #COL stride=2
                for l in range(0, 2): #POOL SIZE 2x2
                    for m in range(0, 2):
                        temp_now = Input[i][l+j][m+k]
                        if(maxpool_out[i][int(j/2)][int(k/2)] < temp_now):
                            maxpool_out[i][int(j/2)][int(k/2)] = temp_now
    return maxpool_out
  
## Fully Connected Layer
def FullyConnected(Input, Weight, Bias):
    ch = len(Weight) #channel
    ch_before = len(Weight[0]) #channel before
    FC_out = np.zeros((ch)) #define output array
    for i in range(0, ch):
        print("You are at Full Connected Layer: Channel", i)
        FC_out[i] = Bias[i] + FC_out[i] 
        for j in range(0, ch_before):
            FC_out[i]= Input[j] * Weight[i][j] + FC_out[i] 
    return FC_out

# VGG16 Inference START
print("1 Layer Convolution")
conv1_out = Conv_1(in_image, conv1_w, conv1_b)
relu1_out = ReLU(conv1_out)
np.save("conv1_out.npy", conv1_out)

print("2nd Layer Convolution")
conv2_out = Conv(conv1_out, conv2_w, conv2_b)
relu2_out = ReLU(conv2_out)
maxpool_2 = Maxpool2d(conv2_out)

print("3rd layer Convolution")
conv3_out = Conv(maxpool_2, conv3_w, conv3_b)
relu3_out = ReLU(conv3_out)
np.save("conv3_out.npy", conv3_out)

print("4th layer Conv")
conv4_out = Conv(relu3_out, conv4_w, conv4_b)
relu4_out = ReLU(conv4_out)
maxpool_4 = Maxpool2d(relu4_out)
np.save("conv4_out.npy", conv4_out)


print("5th layer Conv")
conv5_out = Conv(maxpool_4, conv5_w, conv5_b)
relu5_out = ReLU(conv5_out)
np.save("conv5_out.npy", conv5_out)

print("6th layer Conv")
conv6_out = Conv(relu5_out, conv6_w, conv6_b)
relu6_out = ReLU(conv6_out)
np.save("conv6_out.npy", conv6_out)


print("7th layer Conv")
conv7_out = Conv(relu6_out, conv7_w, conv7_b)
relu7_out = ReLU(conv7_out)
maxpool_7 = Maxpool2d(relu7_out)
np.save("conv7_out.npy", conv7_out)


print("8th layer Conv")
conv8_out = Conv(maxpool_7, conv8_w, conv8_b)
relu8_out = ReLU(conv8_out)
np.save("conv8_out.npy", conv8_out)

print("9th layer Conv")
conv9_out = Conv(relu8_out, conv9_w, conv9_b)
relu9_out = ReLU(conv9_out)
np.save("conv9_out.npy", conv9_out)

print("10th layer Conv")
conv10_out = Conv(relu9_out, conv10_w, conv10_b)
relu10_out = ReLU(conv10_out)
maxpool_10 = Maxpool2d(relu10_out)
np.save("conv10_out.npy", conv10_out)

print("11th layer Conv")
conv11_out = Conv(maxpool_10, conv11_w, conv11_b)
relu11_out = ReLU(conv11_out)
np.save("conv11_out.npy", conv11_out)

print("12th layer Conv")
conv12_out = Conv(relu11_out, conv12_w, conv12_b)
relu12_out = ReLU(conv12_out)
np.save("conv12_out.npy", conv12_out)

print("13th layer Conv")
conv13_out = Conv(relu12_out, conv13_w, conv13_b)
relu13_out = ReLU(conv13_out)
maxpool13 = Maxpool2d(relu13_out)
np.save("conv13_out.npy", conv13_out)

print("13th Conv layer Average Pooling ")
conv13_torch = torch.from_numpy(maxpool13) #numpy array轉torch array
torch_AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(7)
avgpool13 = torch_AdaptiveAvgPool2d(conv13_torch)
fc_input = avgpool13.numpy()

print("14th FC layer")
flattened_input = np.reshape(fc_input, len(fc_input)*len(fc_input[0])*len(fc_input[0][0])) #flatten input for Fully Connected layer
fc14_out = FullyConnected(flattened_input, fc14_w, fc14_b)
fc14_relu = ReLU_FC(fc14_out)
np.save("fc14_out.npy", fc14_out)

print("15th FC layer")
fc15_out = FullyConnected(fc14_relu, fc15_w, fc15_b)
fc15_relu = ReLU_FC(fc15_out)
np.save("fc15_out.npy", fc15_out)

print("16th FC layer")
fc16_out = FullyConnected(fc15_relu, fc16_w, fc16_b)
np.save("fc16_out.npy", fc16_out)
index_of_max= np.argmax(fc16_out)
print("The maximum value = ", fc16_out[index_of_max])

print("The maximum value = ", fc16_out[index_of_max], "It appears at: ", index_of_max)
## Result
fc16_out = np.load('D:\Workspace\pythonwork\VGG16_ML_LAB2\FC16_out.npy')
print(correctout[452])
