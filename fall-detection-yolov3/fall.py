from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from PIL import Image


def parse_Arguement():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "input/29.mp4", type = str)
    
    return parser.parse_args()
    
arg = parse_Arguement()
batchSize = int(arg.bs)
confidence = float(arg.confidence)
nonMax_threshold = float(arg.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

numOfClasses = 80
classes = loadingClasses("data/coco.names")

#Set up the neural network
print("Activating YOLO network.....")
model = Darknet(arg.cfgfile)
model.loadingWeights(arg.weightsfile)
print("YOLO network initilised")

model.networkInfo["height"] = arg.reso
inputDimension = int(model.networkInfo["height"])
assert inputDimension % 32 == 0 
assert inputDimension > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()

def checkFall(w,h):
    """
    Check orienation of the detected bounding box to determine fall
    
    """

    if w > h:
        return True
    else:
        return False

def drawBbox(x, results):
    """
    Display bounding boxes and label on the object
    
    """
    coord1 = tuple(x[1:3].int())
    coord2 = tuple(x[3:5].int())
    top = int(coord1[0]) #top coordinate
    left = int(coord1[1]) #left coordinate
    btm = int(coord2[0]) #bottom coordinate
    right = int(coord2[1]) #right coordinate
    img = results
    cls = int(x[-1])
    i = Image.fromarray(img)
    det_obj = i.crop((top,left,btm,right)) #crop the detected object
    w, h = det_obj.size #Get the dimension of the detected object (width and height)
    color = random.choice(colors)
    label = "{0}".format(classes[cls]) #obtain label of the detected class
    if label =="person":
        if checkFall(w, h):

            cv2.rectangle(img,coord1,coord2,(0,0,255),1)
            txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            coord2 = coord1[0] + txt_size[0] + 3, coord1[1] + txt_size[1] + 4
            cv2.rectangle(img,coord1,coord2,(0,0,255),-1)
            cv2.putText(img, "Fall!",(coord1[0], coord1[1] + txt_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

            #start a time
            #end time
            #count the time 
            #if more than 15 secs then change label into fatal condition.
        else:    
            cv2.rectangle(img, coord1, coord2,(0,255,0), 1) #draw bbox for obj
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            coord2 = coord1[0] + t_size[0] + 3, coord1[1] + t_size[1] + 4
            cv2.rectangle(img, coord1, coord2,(0,255,0), -1)
            cv2.putText(img, label, (coord1[0], coord1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Detection Phase~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inputVideo = arg.videofile #or path to the video file. 

cap = cv2.VideoCapture(inputVideo)  

#cap = cv2.VideoCapture(0)  for webcam

# Get the frame dimension of the video
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))
# Save the output video
out = cv2.VideoWriter('output/fall.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frameWidth,frameHeight));
# Throws error when no video files are loaded
assert cap.isOpened(), 'No video presented' 

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prepare_image(frame, inputDimension)
#        cv2.imshow("a", frame)
        imgDimension = frame.shape[1], frame.shape[0]
        imgDimension = torch.FloatTensor(imgDimension).repeat(1,2)   
                     
        if CUDA:
            imgDimension = imgDimension.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, numOfClasses, nms_conf = nonMax_threshold)


        if type(output) == int:
            frames += 1
            print("Video FPS: {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            #Initiallise the Q key to kill the program
            if key & 0xFF == ord('q'):
                break
            continue
        
        
        

        imgDimension = imgDimension.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/imgDimension,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inputDimension - scaling_factor*imgDimension[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inputDimension - scaling_factor*imgDimension[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, imgDimension[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, imgDimension[i,1])
    
        
        

        classes = loadingClasses('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: drawBbox(x, frame), output))        
        out.write(frame)
        cv2.imshow("Fall detector", frame)
        key = cv2.waitKey(1)
        #Initiallise the Q key to kill the program
        if key & 0xFF == ord('q'):
            break
        frames += 1
        #print(time.time() - start)
        print("Video FPS: {:5.2f}".format( frames / (time.time() - start)))
    else:
        break

end = time.time()
cap.release()
out.release()