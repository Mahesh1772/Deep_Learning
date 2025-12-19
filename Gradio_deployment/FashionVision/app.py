
from model import create_vgg
from typing import Tuple, Dict
from timeit import default_timer as timer
import gradio as gr
import os 
import torch

# Setup class names
fmnist_classes = ['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']

# Model preparation and transfroms
vgg, vgg_transforms = create_vgg()

# Model name
model_name = 'vgg_fmnist_20_percent_5epochs.pth'

# Load the pretrained weights
vgg.load_state_dict(torch.load(f=model_name,
                    map_location=torch.device('cpu'),
                    ))

## Predict Function ##
def predict(image):
  """Transforms and performs a prediction on img and returns prediction and time taken.
  """
  # Start timer 
  start_time = timer()

  # Transfrom the image using VGG architecture
  image = vgg_transforms(image).unsqueeze(0).to(device)

  # Put the model into eval mode and inference
  vgg.eval()
  with torch.inference_mode():
    # Get prediction prediction probabilities
    pred_probs = vgg(image).softmax(dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {fmnist_classes[i]: float(pred_probs[0][i]) for i in range(len(fmnist_classes))}

    # Calculate prediction time
    pred_time = round(timer() - start_time, 4)

  return pred_labels_and_probs, pred_time

## Gradio App ##

# Create title, description and article strings
title = "FashionVision Mini"
description = "An VGG16 feature extractor computer vision model to classify images of clothing."
article = "Done as part of [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=4, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
