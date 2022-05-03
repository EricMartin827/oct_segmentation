################################################################################
################################ Python Imports ################################
################################################################################

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    jaccard_score,
    accuracy_score
)

################################################################################
################################# Local Imports ################################
################################################################################


################################################################################
########################### Main Functions and Classes #########################
################################################################################


def compute_entropy(output):
  '''
  ouput: is a tensor of dimension (B,C,H,W) where each entry
  is the probability of belonging to class c

  output_uncertainty: tensor of dimension (B,1,H,W,) where each
  entry is the entropy as U = \sum^C p_i log(p_i)
  '''
  B,C,H,W = output.shape
  # softmax to convert output to probabilities
  m = torch.nn.Softmax(dim=1)
  output = m(output)
  # compute entropy as uncertainty
  output_uncertainty = torch.sum(torch.mul(output, torch.log(output)), dim=1)
  # print("Check output is coverted to probability:", output[0,:,0,0])
  return -output_uncertainty


def compute_auc(pred, y):
  '''
  pred, y: is a tensor of dimension (B,C,H,W) where each entry
  is the one-hot encoded binary prediction of a class

  accuracy_metric: tensor of dimension (B,C) where each
  entry is the sample-wise and class-wise accuracy
  '''
  B,C,H,W = pred.shape
  # compute accuracy for each sample each class
  auc_metric = torch.zeros((B,C))
  pred_vec = pred.flatten(start_dim=2).cpu()
  y_vec = y.flatten(start_dim=2).cpu()
  for b in range(B):
      for c in range(C):
          auc_metric[b,c] = roc_auc_score(pred_vec[b,c], y_vec[b,c])
  return auc_metric


def compute_jaccard(pred, y):
  '''
  pred, y: is a tensor of dimension (B,C,H,W) where each entry
  is the one-hot encoded binary prediction of a class

  jaccard_metric: tensor of dimension (B,C) where each
  entry is the sample-wise and class-wise jaccard score
  '''
  B,C,H,W = pred.shape
  # compute accuracy for each sample each class
  jaccard_metric = torch.zeros((B,C))
  pred_vec = pred.flatten(start_dim=2).cpu()
  y_vec = y.flatten(start_dim=2).cpu()
  for b in range(B):
      for c in range(C):
          jaccard_metric[b,c] = jaccard_score(pred_vec[b,c], y_vec[b,c])
  return jaccard_metric


def compute_accuracy(pred, y):
  '''
  pred, y: is a tensor of dimension (B,C,H,W) where each entry
  is the one-hot encoded binary prediction of a class

  accuracy_metric: tensor of dimension (B,C) where each
  entry is the sample-wise and class-wise accuracy
  '''
  B,C,H,W = pred.shape
  # flatten lasst two dimensions
  pred_vec = torch.flatten(pred, start_dim=2).cpu()
  y_vec = torch.flatten(y, start_dim=2).cpu()
  # compute accuracy for each sample each class
  accuracy_metric = torch.zeros((B,C))
  for b in range(B):
      for c in range(C):
          accuracy_metric[b,c] = accuracy_score(pred_vec[b,c], y_vec[b,c])
  #accuracy_metric = torch.round(torch.sum(pred_vec == y_vec, dim=2) / (H*W), 4)
  return accuracy_metric

def predict(model, image, K=1):
    '''
    Produces K stochastic forward passes

    model: must be a trained model
    image: data to predict on
    K: number of forward passes HERE

    output: binary predictions (N,C,H,W)
    output_uncertainty: (N,C,H,W)
    '''

    N,_,H,W = image.shape

    if K == 1:
        model.eval()
        output_probs = F.softmax(model(image), dim=1)
    else:
        model.train()
        output_K = torch.zeros(K, N, 3, H, W)
        for k in range(K):
            output_K[k,:,:,:] = F.softmax(model(image), dim=1)
        output_probs = torch.mean(output_K, dim=0)

    # compute uncertainty based on entropy
    output_uncertainty = compute_entropy(output_probs)

    # TODO: alternative way to compute uncertainty is to use
    # variance of the K passes
    return output_probs, output_uncertainty
    
    
# NEW 4/27/2022
def compute_thickness(mask, resolution=1): 
    '''
    Computes the vertical thickness of each image given a batch of masks
    for each class

    mask: is a tensor of dimension (B,C,H,W) where each entry
    OR a list of images
    is the categorical classification with discrete values [0,1,...C-1]

    thickness: a tensor of dimension (B, C) 
    '''
    if isinstance(mask, list): # if a list 
        B = len(mask)
        C = mask[0].shape[0]
        thickness = torch.zeros((B, C))
        # loop through each image
        for b in range(B):
            # sum thickness through the horizontal axis for each class
            for c in range(C):
                thickness[b, int(c)] = torch.mean(torch.sum(mask[b][c,:,:]==1, dim=1).float()) 
    else: # if a tensor
        B,C = mask.shape[0], mask.shape[1]
        thickness = torch.zeros((B, C))
        # loop through each image
        for b in range(B):
            # sum thickness through the horizontal axis for each class
            for c in range(C):
                thickness[b, int(c)] = torch.mean(torch.sum(mask[b,c,:,:]==1, dim=1).float()) 
    return thickness * resolution 
    
    
# NEW 4/27/2022
def compute_pixels(mask, metric="proportion"): 
    '''
    Computes the number of pixels of each image given a batch of masks
    for each class

    mask: is a tensor of dimension (B,C,H,W) where each entry OR a list of images
    values are categorical classification with discrete values [0,1,...C-1]
    metric="proportion" to compute percentage of pixels; else the count 
    pixels: a tensor of dimension (B, C) 
    '''
    if isinstance(mask, list): # if a list 
        B = len(mask)
        C,H,W = mask[0].shape
        pixels = torch.zeros((B, C))
        # loop through each image
        for b in range(B):
            # count number of pixels in the whole image for that class
            for c in range(C):
                pixels[b, int(c)] = torch.sum(mask[b][c,:,:]==1).float() 
    else: # if a tensor
        B = mask.shape[0]
        C,H,W = mask[0,:,:,:].shape
        pixels = torch.zeros((B, C))
        # loop through each image
        for b in range(B):
            # sum thickness through the horizontal axis for each class
            for c in range(C):
                pixels[b, int(c)] = torch.sum(mask[b,c,:,:]==1).float()
    return pixels / (H*W)
    
    
# NEW 4/30/2022
def getGlaucomaLabel(path, duplicate=0):
    names = pd.read_excel(path)
    idx = names.iloc[:,0].str.contains("mask")
    names = names.loc[idx, :]
    glaucoma = np.asarray((names.iloc[:,0].str.contains("ONC") + names.iloc[:,0].str.contains("OHT"))).astype(int)
    if duplicate > 0:
        n = glaucoma.shape[0]
        glaucoma_new = []
        for i in range(n):
    	    for j in range(duplicate):
                glaucoma_new.append(glaucoma[i])
        return np.asarray(glaucoma_new)	
    return glaucoma 
