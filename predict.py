import torch
import argparse
import PIL
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy
from torch.autograd import Variable
import json
from torchvision import models
from torchvision import transforms


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)

    # 1. resize the image
    if im.width > im.height:
        height = 256
        width = int(256 * im.width / im.height)
    else:
        width = 256
        height = int(256 * im.height / im.width)

    im_resized = im.resize((width, height))
    # then central crop a 224x224
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2

    im_resized = im_resized.crop((left, top, right, bottom))

    # 2. update the color channel, normalize as 0-1
    np_img = np.array(im_resized)
    np_img = np_img / 255

    # 3. normalize data
    loader = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    np_img = loader(torch.from_numpy(np_img)).numpy()

    # 5. update the color channel
    np_img = np_img.transpose(2, 0, 1)

    # Final convertion
    torch_img = torch.from_numpy(np_img)
    torch_img = torch_img.type(torch.FloatTensor)
    return torch_img


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # load the image
    img = process_image(image_path)
    # process the data, send to GPU and make a fake batch dimension
    img = Variable(img, requires_grad=False)
    img = img.unsqueeze(0)
    img = img.to(device)

    # through model
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class


parser = argparse.ArgumentParser(
    description='This program will inference with pre-trained model.'
)

parser.add_argument('path_img',
                    action="store",
                    help="Path to the image for predict.")
parser.add_argument('path_chk',
                    action="store",
                    help="Path to the checkpoint for predict.")
parser.add_argument('--category_names',
                    action="store",
                    dest='cat_names',
                    default="cat_to_name.json",
                    help="Input saved checkpoint path. [default: cat_to_name.json]")
parser.add_argument('--arch',
                    action="store",
                    dest='arch',
                    default="vgg11",
                    help="Input for pre-trained model type. [default: vgg11]")
parser.add_argument('--gpu',
                    action="store_true",
                    dest='gpu',
                    default="False",
                    help="Input whether use gpu. [default: False]")
parser.add_argument('--top_k',
                    action="store_true",
                    dest='top_k_num',
                    default=5,
                    help="Input top probability result. [default: 5]")

results = parser.parse_args()

# First recreate the model
if results.arch == "vgg16":
    model = models.vgg16(pretrained=True)
elif results.arch == "vgg13":
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg11(pretrained=True)

checkpoint_load = torch.load(results.path_chk)

model.classifier = checkpoint_load['classifier']
model.load_state_dict(checkpoint_load['state_dict'])
model.classifier.load_state_dict(checkpoint_load['classifier.state_dict'])
model.class_to_idx = checkpoint_load['class_to_idx']

# Final checks before predict.
if results.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)

# inform the result
print("The predict will be run in: ", device)

# run prediction
probs, classes = predict(results.path_img, model, results.top_k_num)

if results.gpu == True:
    probs_cpu = probs.cpu()
    classes_cpu = classes.cpu()
else:
    probs_cpu = probs
    classes_cpu = classes

# Load JSON catagory data
with open(results.cat_names, 'r') as f:
    cat_to_name = json.load(f)
# Invers the lookup table
inv_map = {v: k for k, v in model.class_to_idx.items()}

class_name = []
for i in np.nditer(classes_cpu.numpy()):
    class_name.append(cat_to_name[inv_map.get(int(i))])

print("The most possible results (in order) are: ")
for p in class_name: print(p)
print("Responding probabilities are: ")
for p in probs_cpu.numpy().tolist()[0]: print(p)