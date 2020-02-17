import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms, datasets

parser = argparse.ArgumentParser(
    description='This program will train the network.'
)

parser.add_argument('data_pth',
                    action="store",
                    help="Path to the data for training.")
parser.add_argument('--learning_rate',
                    action="store",
                    dest='lr',
                    default=0.003,
                    help="Input learning rate.")
parser.add_argument('--hidden_units',
                    action="store",
                    dest='hidden_units',
                    default=512,
                    type=int,
                    help="Input number of hiden units within hidden layer. [default: 512]")
parser.add_argument('--epochs',
                    action="store",
                    dest='epoch',
                    default=1,
                    type=int,
                    help="Input target training epoch. [default: 1]")
parser.add_argument('--save_dir',
                    action="store",
                    dest='dir',
                    default="by_check.pth",
                    help="Input saved checkpoint path. [default: ./train_py_checkpoint.pth]")
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

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
results = parser.parse_args()

## Loading data
train_dataset = datasets.ImageFolder(results.data_pth, transform=train_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

## Create model and laod the pre-trained model
if results.arch == "vgg16":
    model = models.vgg16(pretrained=True)
elif results.arch == "vgg13":
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg11(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# The is a 25088 in and 1000 out full connected network. Need to replace to match our application.
# In our application, its a 25088 in and 102 out network
ClassifierNew = nn.Sequential(nn.Linear(25088, results.hidden_units),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(results.hidden_units, results.hidden_units),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(results.hidden_units, 102),
                              nn.LogSoftmax(dim=1))

model.classifier = ClassifierNew

# Now create a loss function and a optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=results.lr)
criterion = nn.NLLLoss()

# Final checks before training.
if results.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)

# inform the result
print("The training will be run in: ", device)

# Start training
epoch = results.epoch
running_loss = 0

for e in range(epoch):
    for images, labels in train_dataloader:
        # Send data to device
        images, labels = images.to(device), labels.to(device)
        # Flatten the image

        optimizer.zero_grad()

        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    else:
        # For each epoch, print the result
        print("Epoch: ", e + 1, "/", epoch)
        print("  Loss for train: ", running_loss / len(train_dataloader))
        # Reset statistic for next epoch
        running_loss = 0

# training finish. Now save result and exit
model.class_to_idx = train_dataset.class_to_idx
# Construct dic to preserve information
checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'classifier.state_dict': model.classifier.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict()}
# Save
print("Trained checkpoint saved in ", results.dir)
torch.save(checkpoint, results.dir)
