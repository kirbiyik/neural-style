import uuid
from collections import namedtuple
import gc

from skimage import color 
import numpy as np

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

import PIL
import torch
import torch.nn as nn
from scipy.misc import imread

import torchvision
import torchvision.transforms as T


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    vnum = int(scipy.__version__.split('.')[1])
    major_vnum = int(scipy.__version__.split('.')[0])
    
    assert vnum >= 16 or major_vnum >= 1, "You must install SciPy >= 0.16.0 to complete this notebook."


def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.
    
    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.
    
    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    if torch.cuda.is_available():
        x = x.cuda()
    features = []
    prev_feat = x
    for _, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    """
    channels = content_current.size(1)
    content_current = content_current.view(channels, -1)
    channels = content_original.size(1)
    content_original = content_original.view(channels, -1)
    loss = content_weight * ((content_current - content_original)**2).sum()
    return loss
    

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N, C, H, W = features.shape
    features = features.view(N, C, -1)
    gram = torch.bmm(features, features.permute([0,2,1]))
    if normalize:
        gram = gram / (C * H * W)
    return gram 


    
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    losses = []
    for i, layer in enumerate(style_layers):
        losses.append(style_weights[i] * \
                ((gram_matrix(feats[layer]) - style_targets[i]) ** 2).sum())
    style_loss = sum(losses)
    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    N, C, H, W = img.size()
    h_var = img[:,:, torch.arange(1, H).long(), :] - img[:,:, torch.arange(0, H-1).long(),:]
    w_var = img[:,:,:, torch.arange(1, W).long()] - img[:,:,:, torch.arange(0, W-1).long()]
    h_var, w_var = (h_var ** 2).sum(), (w_var ** 2).sum()
    return tv_weight * (h_var + w_var)
    

def style_transfer(cnn, content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
    """
    Run style transfer!
    
    Inputs:
    - cnn: CNN model to extracf features
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    # Extract features for the content image
    pil_img = PIL.Image.open(content_image)
    # Some png files have alpha channel, esp. screenshots. Convert them to rgb
    if pil_img.mode == 'RGBA':
        pil_img = color.rgba2rgb(pil_img)
        pil_img = PIL.Image.fromarray(pil_img.astype('uint8'))
    content_img = preprocess(pil_img, size=image_size)
    feats = extract_features(content_img, cnn)
    content_target = feats[content_layer].clone()
    # Extract features for the style image
    pil_img = PIL.Image.open(style_image)
    if pil_img.mode == 'RGBA':
        pil_img = color.rgba2rgb(pil_img)
        pil_img = PIL.Image.fromarray(pil_img.astype('uint8'))
    style_img = preprocess(pil_img, size=style_size)

    feats = extract_features(style_img, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img.requires_grad_()
    if torch.cuda.is_available():
        img = img.cuda()
    
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img Torch tensor, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img], lr=initial_lr)
    
    # f, axarr = plt.subplots(1,2)
    # axarr[0].axis('off')
    # axarr[1].axis('off')
    # axarr[0].set_title('Content Source Img.')
    # axarr[1].set_title('Style Source Img.')
    # axarr[0].imshow(deprocess(content_img.cpu()))
    # axarr[1].imshow(deprocess(style_img.cpu()))
    # plt.show()
    # plt.figure()
    
    for t in range(250):
        if t < 190:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(img, cnn)
        
        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img, tv_weight) 
        loss = c_loss + s_loss + t_loss
        
        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img], lr=decayed_lr)
        optimizer.step()
        
        # if t % 100 == 0:
        #     print('Iteration {}'.format(t))
        #     plt.axis('off')
        #     plt.imshow(deprocess(img.data.cpu()))
        #     plt.show()
    #print('Iteration {}'.format(t))
    #plt.axis('off')
    #plt.imshow(deprocess(img.data.cpu()))
    unique_name = str(uuid.uuid4()) + ".png"
    result_path = "static/images/"+ unique_name
    plt.imsave(result_path, deprocess(img.detach().cpu()))
    return unique_name
    


def main(content_image_path, style_image_path):
    content_image_path  = 'static/images/' + content_image_path
    style_image_path = 'static/images/' + style_image_path
    check_scipy()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    cnn = torchvision.models.squeezenet1_1(pretrained=True).features
    cnn.type(dtype)
    # disable autograd
    for param in cnn.parameters():
        param.requires_grad = False

    # Starry Night + Tubingen
    params = {
        'cnn': cnn,
        'content_image' : content_image_path,
        'style_image' : style_image_path,
        'image_size' : 192,
        'style_size' : 192,
        'content_layer' : 3,
        'content_weight' : 2e-2,
        'style_layers' : [1, 4, 6, 7],
        'style_weights' : [300000, 1500, 15, 3],
        'tv_weight' : 2e-1
    }
    # <unique_filename>.png
    f_name = style_transfer(**params)
    gc.collect()
    return f_name
