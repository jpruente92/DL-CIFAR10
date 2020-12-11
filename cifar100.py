from torchvision import models
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from use_pretrained_model import use_pretrained_model, Network


# hyperparameters

num_epochs = 25
batch_size = 128
learning_rate = 0.001
momentum = 0.9
print_every = 10
hidden_layers = [512]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frozen = False # whether the parameters in the loaded model are trained or not


# dataset
    # dataset has PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)

classes=train_dataset.classes



print("RESNET NOT FROZEN")
model =torchvision.models.resnet18(pretrained=True)
# Freeze parameters so we don't backprop through them
if frozen:
    for param in model.parameters():
        param.requires_grad = False
input_size_last_layer = 512 # has to be adapted according to model
model.fc=Network(input_size_last_layer,len(classes),hidden_layers) # has to be adapted according to model

acc_resnet=use_pretrained_model(model,classes,learning_rate, momentum,train_dataset, test_dataset,batch_size,device,num_epochs,print_every)

print("DENSENET NOT FROZEN")
model =model = models.densenet121(pretrained=True)
# Freeze parameters so we don't backprop through them
if frozen:
    for param in model.parameters():
        param.requires_grad = False
input_size_last_layer = 1024 # has to be adapted according to model
model.classifier=Network(input_size_last_layer,len(classes),hidden_layers) # has to be adapted according to model

acc_densenet=use_pretrained_model(model,classes,learning_rate, momentum,train_dataset, test_dataset,batch_size,device,num_epochs,print_every)

print("VGG NOT FROZEN")
model =model = models.vgg16(pretrained=True)
# Freeze parameters so we don't backprop through them
if frozen:
    for param in model.parameters():
        param.requires_grad = False
input_size_last_layer = 4096 # has to be adapted according to model
model.classifier[6]=Network(input_size_last_layer,len(classes),[2048,1024,512]) # has to be adapted according to model

acc_vgg=use_pretrained_model(model,classes,learning_rate, momentum,train_dataset, test_dataset,batch_size,device,num_epochs,print_every)

print("RESNEXT NOT FROZEN")
model =model =torchvision.models.resnext50_32x4d(pretrained=True)
# Freeze parameters so we don't backprop through them
if frozen:
    for param in model.parameters():
        param.requires_grad = False
input_size_last_layer = 2048 # has to be adapted according to model
model.fc=Network(input_size_last_layer,len(classes),[1024,512]) # has to be adapted according to model

acc_resnext=use_pretrained_model(model,classes,learning_rate, momentum,train_dataset, test_dataset,batch_size,device,num_epochs,print_every)



# create plot
n_groups=11
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index, acc_resnet[:11], bar_width,
alpha=opacity,
color='b',
label='ResNet')

rects2 = plt.bar(index + bar_width, acc_densenet[:11], bar_width,
alpha=opacity,
color='g',
label='DenseNet')

rects3 = plt.bar(index + 2*bar_width, acc_vgg[:11], bar_width,
alpha=opacity,
color='r',
label='VGG')

rects3 = plt.bar(index + 3*bar_width, acc_resnext[:11], bar_width,
alpha=opacity,
color='y',
label='ResNext')


label_names =["Total"]
for c in classes[:10]:
    label_names.append(c)

plt.xlabel('Network')
plt.ylabel('Accuracies')
plt.title('CIFAR 100')
plt.xticks(index + bar_width, label_names)
plt.legend()

plt.savefig("cifar100.png")

plt.show()



