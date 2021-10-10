### Source from
### https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt

# Initialize the model
#model = torch.load('<PATH_FILE_NAME>.pth')
model = torchvision.models.resnet18(pretrained=False)


# Set the model to run on the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model on Eval Mode
model.eval()

from torchvision import transforms
from PIL import Image

# Open the image file
image = Image.open('Vanilla_backprop_1.jpg')

# Set up the transformations
transform_ = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),transforms.ToTensor(),
		transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])

# Transforms the image
image = transform_(image)

image = image.reshape(1, 3, 224, 224)

image = image.to(device)

image.requires_grad_()

# Retrieve output from the image
output = model(image)
print(output.shape)

# Catch the output
output_idx = output.argmax()
print(output_idx)

output_max = output[0, output_idx]
print(output_max)

# Do backpropagation to get the derivative of the output based on the image
output_max.backward()

# Retireve the saliency map and also pick the maximum value from channels on each pixel.
# In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)

saliency, _ = torch.max(image.grad.abs(), dim=1)
print(saliency.shape)

saliency = saliency.reshape(224, 224)

# Reshape the image
image = image.reshape(-1, 224, 224)

# Visualize the image and the saliency map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
ax[0].axis('off')
ax[1].imshow(saliency.cpu(), cmap='hot')
ax[1].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()