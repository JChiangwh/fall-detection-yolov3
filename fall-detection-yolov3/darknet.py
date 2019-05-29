from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 


def configParse(cfgfile):
    """
    Takes in the configuration file
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-initilise the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks


class EmptyLayer(nn.Module):
    """
    Empty layer initialised for Route and Shortcut layer to bypass the layer operations process.

    """
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
class yoloDetectionLayer(nn.Module):
    """
    Detection layer initialised for YOLO layer.

    """
    def __init__(self, anchors):
        super(yoloDetectionLayer, self).__init__()
        self.anchors = anchors

def constructModule(blocks):
    """
    Construction of the YOLOv3 model using PyTorch library.

    """
    networkInfo = blocks[0]     #Captures the information about the input and pre-processing    
    moduleList = nn.ModuleList()
    previousFilters = 3
    outputFilters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to moduleList
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batchNorm = int(x["batch_normalize"])
                bias = False
            except:
                batchNorm = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernelSize = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernelSize - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            convolutional = nn.Conv2d(previousFilters, filters, kernelSize, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), convolutional)
        
            #Add the Batch Norm Layer
            if batchNorm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            if activation == "leaky":
                activationFunction = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activationFunction)
        
            #If it's an upsampling layer
            #Use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = outputFilters[index + start] + outputFilters[index + end]
            else:
                filters= outputFilters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            yolo_detection = yoloDetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), yolo_detection)
                              
        moduleList.append(module)
        previousFilters = filters
        outputFilters.append(filters)
        
    return (networkInfo, moduleList)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = configParse(cfgfile)
        self.networkInfo, self.moduleList = constructModule(self.blocks)
        
    def forward(self, x, CUDA):
        """
        Function for the feed forward process of the network

        """
        modules = self.blocks[1:]
        outputs = {}   #Cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            moduleType = (module["type"])
            
            if moduleType == "convolutional" or moduleType == "upsample":	#Check for convolutional and upsample type
                x = self.moduleList[i](x)
    
            elif moduleType == "route":										#Check for route layer
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    featureMap1 = outputs[i + layers[0]]
                    featureMap2 = outputs[i + layers[1]]
                    x = torch.cat((featureMap1, featureMap2), 1)
                
    
            elif  moduleType == "shortcut":									#Check for shortcut layer
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif moduleType == 'yolo':        								#Check for yolo layer
                anchors = self.moduleList[i][0].anchors
                #Get the input dimensions
                inputDimension = int (self.networkInfo["height"])
        
                #Get the number of classes
                numOfClasses = int (module["classes"])
        
                #Transform 
                x = x.data
                x = transformPrediction(x, inputDimension, anchors, numOfClasses, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections


    def loadingWeights(self, weightfile):
        """
        Loading weights file into the network

        """

        #Open the weights file
        initialiseWeights = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(initialiseWeights, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(initialiseWeights, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.moduleList)):
            moduleType = self.blocks[i + 1]["type"]
    
            #If module type is convolutional load weights
            #Otherwise ignote it.
            
            if moduleType == "convolutional":
                model = self.moduleList[i]
                try:
                    batchNorm = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batchNorm = 0
            
                convolutional = model[0]
                
                
                if (batchNorm):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    bn_weights_num = bn.bias.numel()
        
                    #Load the weights
                    bias_bn = torch.from_numpy(weights[ptr:ptr + bn_weights_num])
                    ptr += bn_weights_num
        
                    weights_bn = torch.from_numpy(weights[ptr: ptr + bn_weights_num])
                    ptr  += bn_weights_num
        
                    runningMean_bn = torch.from_numpy(weights[ptr: ptr + bn_weights_num])
                    ptr  += bn_weights_num
        
                    runningVar_bn = torch.from_numpy(weights[ptr: ptr + bn_weights_num])
                    ptr  += bn_weights_num
        
                    #Cast the loaded weights into dims of model weights. 
                    bias_bn = bias_bn.view_as(bn.bias.data)
                    weights_bn = weights_bn.view_as(bn.weight.data)
                    runningMean_bn = runningMean_bn.view_as(bn.running_mean)
                    runningVar_bn = runningVar_bn.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bias_bn)
                    bn.weight.data.copy_(weights_bn)
                    bn.running_mean.copy_(runningMean_bn)
                    bn.running_var.copy_(runningVar_bn)
                
                else:
                    #Number of biases
                    numOfBiases = convolutional.bias.numel()
                
                    #Load the weights
                    convolution_bias = torch.from_numpy(weights[ptr: ptr + numOfBiases])
                    ptr = ptr + numOfBiases
                
                    #reshape the loaded weights according to the dims of the model weights
                    convolution_bias = convolution_bias.view_as(convolutional.bias.data)
                
                    #Finally copy the data
                    convolutional.bias.data.copy_(convolution_bias)
                    
                #Load the weights for the Convolutional layers
                numOfWeights = convolutional.weight.numel()
                
                #Do the same as above for weights
                convolution_weights = torch.from_numpy(weights[ptr:ptr+numOfWeights])
                ptr = ptr + numOfWeights
                
                convolution_weights = convolution_weights.view_as(convolutional.weight.data)
                convolutional.weight.data.copy_(convolution_weights)


