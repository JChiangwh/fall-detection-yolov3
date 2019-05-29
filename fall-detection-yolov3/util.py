from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def prepare_image(img, inputDimension):
    """
    Prepare image to feed it into the neural network. 
    
    """
    img = (letterbox_image(img, (inputDimension, inputDimension)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def loadingClasses(namesfile):
    """
    Loading the class file.

    """
    initialiseClasses = open(namesfile, "r")
    names = initialiseClasses.read().split("\n")[:-1]
    return names

def transformPrediction(prediction, inputDimension, anchors, numOfClasses, CUDA = True):
    """
    Transform the predicted feature map into thhe same size 
    Return transformed prediction
    
    """
    batchSize = prediction.size(0)
    stride =  inputDimension // prediction.size(2)
    gridSize = inputDimension // stride
    attributesOfBbox = 5 + numOfClasses
    numOfAnchors = len(anchors)
    
    prediction = prediction.view(batchSize, attributesOfBbox*numOfAnchors, gridSize*gridSize)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batchSize, gridSize*gridSize*numOfAnchors, attributesOfBbox)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(gridSize)
    a,b = np.meshgrid(grid, grid)

    offset_x = torch.FloatTensor(a).view(-1,1)
    offset_y = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        offset_x = offset_x.cuda()
        offset_y = offset_y.cuda()

    offset_x_y = torch.cat((offset_x, offset_y), 1).repeat(1,numOfAnchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += offset_x_y

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(gridSize*gridSize, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + numOfClasses] = torch.sigmoid((prediction[:,:, 5 : 5 + numOfClasses]))

    prediction[:,:,:4] *= stride
    
    return prediction

def iouOfBbox(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    """
    #Get the coordinates of bounding boxes
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    coord_inter_x1 =  torch.max(box1_x1, box2_x1)
    coord_inter_y1 =  torch.max(box1_y1, box2_y1)
    coord_inter_x2 =  torch.min(box1_x2, box2_x2)
    coord_inter_y2 =  torch.min(box1_y2, box2_y2)
    
    #Intersection area
    areaOfIntersection = torch.clamp(coord_inter_x2 - coord_inter_x1 + 1, min=0) * torch.clamp(coord_inter_y2 - coord_inter_y1 + 1, min=0)

    #Union Area
    box1_area = (box1_x2 - box1_x1 + 1)*(box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1)*(box2_y2 - box2_y1 + 1)
    
    iou = areaOfIntersection / (box1_area + box2_area - areaOfIntersection)
    
    return iou

def unique(tensor):
    """
    get the classes for a given image
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, numOfClasses, nms_conf = 0.4):
    """
    This function takes the following inputs: predictions, confidence,classes_num,nms_conf
    """
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batchSize = prediction.size(0)

    write = False
    


    for ind in range(batchSize):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ numOfClasses], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = iouOfBbox(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inputDimension):
    """
    Resize image with unchanged aspect ratio using padding
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inputDimension
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inputDimension[1], inputDimension[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
