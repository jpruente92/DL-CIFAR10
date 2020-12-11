from train_and_acc import train_network, compute_accuracy
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def use_pretrained_model(model,classes,learning_rate, momentum,train_dataset, test_dataset,batch_size,device,
                         num_epochs=50,print_every=10):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)


    # optimizer and loss criterion
    loss_criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)

    # start training routine
    train_network(model, optimizer, loss_criterion, train_loader,test_loader, classes, device, num_epochs,print_every,plot_training_scores=False)

    acc, acc_classes= compute_accuracy(model, test_loader, classes, device)
    print("Total accuracy :",acc)
    for i,class_name in enumerate(classes):
        print("\tAccuracy ",class_name+" :",acc_classes[i])

    list =[]
    list.append(acc)
    for a in acc_classes:
        list.append(a)
    return list






class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)


    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)
