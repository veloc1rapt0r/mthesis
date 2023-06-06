# Implementation: Oleh Bakumenko, Univerity of Duisburg-Essen
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import numpy as np
from utility import utils as uu

def plot_image_model(tensor_image,tensor_target,model,device,cmap = "bone"):
    # for UNet, plots as different figures
    # image
    # target liver
    # target cancer
    # perdiction background
    # prediction liver
    # prediction cancer
    # we need an utility funcion to switch dimensions from [Channel*Height*Width] to [Height*Width*Channel] and convert tensor to numpy array

    tensor_image = tensor_image.to('cpu')
    array_image = uu.convert_tensor_to_opencv_array(tensor_image)
    plt.figure()
    plt.imshow(array_image, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Image")
    plt.show()

    array_image_target = uu.convert_tensor_to_opencv_array(tensor_target[0])
    plt.figure()
    plt.imshow(array_image_target, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver Mask")
    plt.show()

    array_image_target = uu.convert_tensor_to_opencv_array(tensor_target[1])
    plt.figure()
    plt.imshow(array_image_target, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer Mask")
    plt.show()

    tensor_image = tensor_image.to(device)
    tensor_prediction = model(tensor_image.unsqueeze(0))
    tensor_prediction = tensor_prediction.to('cpu')
    tensor_prediction = torch.squeeze(tensor_prediction, 0)
    tensor_prediction_bg = tensor_prediction[0, :].unsqueeze(0)
    tensor_prediction_liver = tensor_prediction[1, :].unsqueeze(0)
    tensor_prediction_canser = tensor_prediction[2, :].unsqueeze(0)

    array__prediction_bg = uu.convert_tensor_to_opencv_array(tensor_prediction_bg)
    plt.figure()
    plt.imshow(array__prediction_bg, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Background Pred")
    plt.show()

    array__prediction_liver = uu.convert_tensor_to_opencv_array(tensor_prediction_liver)
    plt.figure()
    plt.imshow(array__prediction_liver, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver Pred")
    plt.show()

    array__prediction_cancer = uu.convert_tensor_to_opencv_array(tensor_prediction_canser)
    plt.figure()
    plt.imshow(array__prediction_cancer, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer Pred")
    plt.show()

def plotImg(tensor_image_in, cmap = 'bone'):
    # a simple function to plot an image from torch.tensor
    # we need an utility funcion to switch dimentions from [channel*Height*Width] to [Height*Width*Channel]
    # and convert tensor to numpy array
    tensor_image = tensor_image_in.to('cpu')
    array_image = uu.convert_tensor_to_opencv_array(tensor_image)
    plt.figure()
    plt.grid(False)
    plt.imshow(array_image, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Image")
    plt.show()

def plot_image_model_subplot_no_bg(tensor_image,tensor_target,model, device):
    # for UNet, plots as subplot
    #   [image                 target liver       target cancer       ]
    #   [                       prediction liver   prediction cancer  ]
    # Input:    tensor_image:Tensor
    #           tensor_target: List(Tensor)
    #           model: UNet
    # we need an utility funcion to switch dimensions from [Channel*Height*Width] to [Height*Width*Channel]
    # and convert tensor to numpy array
    plt.grid(False)
    tensor_image = tensor_image.to('cpu')
    array_image = uu.convert_tensor_to_opencv_array(tensor_image)
    plt.subplot(2, 3, 1)
    plt.imshow(array_image, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Image")

    tensor_image = tensor_image.to(device)
    tensor_prediction = model(tensor_image.unsqueeze(0))
    tensor_prediction = tensor_prediction.to('cpu')
    tensor_prediction = torch.squeeze(tensor_prediction,0)
    tensor_prediction = tensor_prediction[1,:].unsqueeze(0)

    array_prediction = uu.convert_tensor_to_opencv_array(tensor_prediction)
    plt.subplot(2, 3, 5)
    plt.imshow(array_prediction, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver Pred")

    array_image_target = uu.convert_tensor_to_opencv_array(tensor_target[0])
    plt.subplot(2, 3, 2)
    plt.imshow(array_image_target, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver Mask")

    tensor_image = tensor_image.to(device)
    tensor_prediction = model(tensor_image.unsqueeze(0))
    tensor_prediction = tensor_prediction.to('cpu')
    tensor_prediction = torch.squeeze(tensor_prediction,0)
    tensor_prediction = tensor_prediction[2,:].unsqueeze(0)

    array__prediction = uu.convert_tensor_to_opencv_array(tensor_prediction)
    plt.subplot(2, 3, 6)
    plt.imshow(array__prediction, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer Pred")

    array_image_target = uu.convert_tensor_to_opencv_array(tensor_target[1])
    plt.subplot(2, 3, 3)
    plt.imshow(array_image_target, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer Mask")
    plt.show()

def plot_image_model_subplot(tensor_image,tensor_target,model,device,cmap = "bone"):
    # for UNet, plots as subplot
    #   [image                 target liver       target cancer      ]
    #   [prediction background prediction liver   prediction cancer  ]
    # Input:    tensor_image: Tensor
    #           tensor_target: List(Tensor)
    #           model: UNet
    tensor_image = tensor_image.to('cpu')
    array_image = uu.convert_tensor_to_opencv_array(tensor_image)
    plt.grid(False)
    plt.subplot(2, 3, 1)
    plt.imshow(array_image, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Image")

    array_image_target = uu.convert_tensor_to_opencv_array(tensor_target[0])
    plt.subplot(2, 3, 2)
    plt.imshow(array_image_target, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver Mask")

    array_image_target = uu.convert_tensor_to_opencv_array(tensor_target[1])
    plt.subplot(2, 3, 3)
    plt.imshow(array_image_target, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer Mask")

    tensor_image = tensor_image.to(device)
    tensor_prediction = model(tensor_image.unsqueeze(0))
    tensor_prediction = tensor_prediction.to('cpu')
    tensor_prediction = torch.squeeze(tensor_prediction, 0)
    tensor_prediction_bg = tensor_prediction[0, :].unsqueeze(0)
    tensor_prediction_liver = tensor_prediction[1, :].unsqueeze(0)
    tensor_prediction_canser = tensor_prediction[2, :].unsqueeze(0)

    array__prediction_bg = uu.convert_tensor_to_opencv_array(tensor_prediction_bg)
    plt.subplot(2, 3, 4)
    plt.imshow(array__prediction_bg, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Background Pred")

    array__prediction_liver = uu.convert_tensor_to_opencv_array(tensor_prediction_liver)
    plt.subplot(2, 3, 5)
    plt.imshow(array__prediction_liver, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver Pred")

    array__prediction_cancer = uu.convert_tensor_to_opencv_array(tensor_prediction_canser)
    plt.subplot(2, 3, 6)
    plt.imshow(array__prediction_cancer, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer Pred")
    plt.show()


def plot_Img_Targets(tensor_image_in, targets, cmap = 'bone'):
    '''
    Plot a subplot:     [image  target liver    target cancer]
    :param tensor_image_in:
    :param targets:
    :param cmap:
    :return:
    '''
    plt.subplot(1, 3, 1)
    tensor_image = tensor_image_in.to('cpu')
    array_image = uu.convert_tensor_to_opencv_array(tensor_image)
    plt.imshow(array_image, cmap=cmap)
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Image")

    plt.subplot(1, 3, 2)
    array_image_target = uu.convert_tensor_to_opencv_array(targets[0])
    plt.imshow(array_image_target, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Liver target")

    plt.subplot(1, 3, 3)
    array_image_target = uu.convert_tensor_to_opencv_array(targets[1])
    plt.imshow(array_image_target, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.title("Cancer target")
    plt.show()

def plot_Img_Contour(tensor_image_in, targets):
    '''
    Plots the input image and target contours
    :param tensor_image_in:
    :param targets:
    '''

    tensor_image = tensor_image_in.to('cpu')
    liver = np.array(targets[0].squeeze().detach().numpy(), dtype = np.uint8)
    tumors = np.array(targets[1].squeeze().detach().numpy(), dtype = np.uint8)
    array_image = uu.convert_tensor_to_opencv_array(tensor_image)

    X, Y = np.meshgrid(np.arange(256), np.arange(256))
    plt.figure()
    plt.imshow(array_image, cmap = "bone")
    plt.xlim((0, 256))
    plt.ylim((0, 256))
    plt.contour(X, Y, liver, colors = "g", alpha = 0.25, linewidths = 0.5)
    plt.contour(X, Y, tumors, colors = "r", alpha = 0.25, linewidths = 0.5)

def plot_Img_Prediction(tensor_image_in, targets_list, target_class, model_class, model_seg, index, save = False, cmap = 'bone', figsize = (18,7)):
    # plots as subplot
    # [       image                       image with            segmentation output liver    segmentation output cancer]
    # [ classification output    contours for liver and cancer                                                         ]
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    softmax1 = nn.Softmax(dim=1)
    softmax0 = nn.Softmax(dim=0)
    tensor_image_in = tensor_image_in.to(device)
    model_class = model_class.to(device)
    model_seg = model_seg.to(device)
    fig, ax = plt.subplots(1,4,figsize = figsize)
    tensor_image = tensor_image_in.to(device)
    array_image = uu.convert_tensor_to_opencv_array(tensor_image.to('cpu'))
    ax[0].imshow(array_image, cmap=cmap)
    ax[0].set_title("Image nr. " + str(index))
    out = model_class(tensor_image.unsqueeze(0)).squeeze(0)
    out_prob= softmax0(out)
    out_argmax_np = torch.argmax(out_prob).detach().cpu().numpy()
    ax[0].text(0.5, -0.2, f"Target: {target_class}, model output: {np.array2string(np.round(out.detach().cpu().numpy(),2))}, \n model output probability {np.array2string(np.round(100*out_prob.detach().cpu().numpy(),1))}.", fontsize=10, ha="center", transform=ax[0].transAxes)

    for i in range(4):
        ax[i].grid(False)
        ax[i].set_xlim((0, 256))
        ax[i].set_ylim((0, 256))

    liver = np.array(targets_list[0].squeeze().detach().numpy(), dtype = np.uint8)
    tumors = np.array(targets_list[1].squeeze().detach().numpy(), dtype = np.uint8)

    X, Y = np.meshgrid(np.arange(256), np.arange(256))
    ax[1].imshow(array_image, cmap = cmap)
    ax[1].contour(X, Y, liver, colors = "g", alpha = 0.25, linewidths = 0.5)
    ax[1].contour(X, Y, tumors, colors = "r", alpha = 0.25, linewidths = 0.5)
    ax[1].set_title("Segmentation")

    tensor_prediction = model_seg(tensor_image_in.unsqueeze(0))
    tensor_prediction = softmax1(tensor_prediction)
    tensor_prediction = torch.squeeze(tensor_prediction, 0)
    tensor_prediction_liver = tensor_prediction[1, :].unsqueeze(0)
    tensor_prediction_canser = tensor_prediction[2, :].unsqueeze(0)

    array_prediction_liver = uu.convert_tensor_to_opencv_array(tensor_prediction_liver.to('cpu'))
    ax[2].imshow(array_prediction_liver, cmap=cmap)
    ax[2].set_title("Liver Prediction")

    array_prediction_cancer = uu.convert_tensor_to_opencv_array(tensor_prediction_canser.to('cpu'))
    ax[3].imshow(array_prediction_cancer, cmap=cmap)
    ax[3].set_title("Cancer Pred")
    if save:
        plt.savefig(fname = f"output_{str(index)}_target_{str(target_class.detach().numpy())}_pred_{str(out_argmax_np)}.png", bbox_inches='tight')
    plt.tight_layout()
    plt.show()


