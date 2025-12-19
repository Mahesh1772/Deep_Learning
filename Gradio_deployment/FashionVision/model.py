import torch 
import torchvision
from torch import nn

def create_vgg():
  vgg_weights = torchvision.models.VGG16_Weights
  vgg_transforms = torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
  vgg = torchvision.models.vgg16(vgg_weights)

  for params in vgg.parameters():
    params.requires_grad = False

  vgg.classifier[6] = nn.Linear(4096, 10)

  return vgg, vgg_transforms
