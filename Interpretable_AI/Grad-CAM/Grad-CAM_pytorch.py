# Source from https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
# Re-developed by JaeYoung

import cv2
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

LABELS_file = 'imagenet-simple-labels.json'

# VGG-19
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)

# initialize the VGG model
vgg = VGG()

# set the evaluation mode
vgg.eval()

# get the image and use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

img = Image.open('data/shark.jpeg')
img = transform(img).unsqueeze(0)

# get the most likely prediction of the model
pred = vgg(img)
pred_convert = pred.data.squeeze()

# load the imagenet category list
with open(LABELS_file) as f:
    classes = json.load(f)

probs, idx = pred_convert.sort(0, True)
probs = probs.detach().numpy()
idx = idx.numpy()
max_idx = idx[0] # Get the high-probability predicted class idx

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# get the gradient of the output with respect to the parameters of the model
pred[:, max_idx].backward()

# pull the gradients out of the model
gradients = vgg.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = vgg.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())

img = cv2.imread('data/shark.jpeg')
height, width, _ = img.shape
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(cv2.resize(heatmap, (width, height)), cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.3 + img * 0.3
cv2.imwrite('./Grad_CAM.jpg', superimposed_img)