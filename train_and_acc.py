import time
import numpy as np

import torch
import matplotlib.pyplot as plt

# # examples
# loss_criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train_network(model, optimizer, loss_criterion, train_loader,test_loader, classes, device, num_epochs,print_every,compute_accuracy_at_epoch=False, plot_training_scores=True):
    model.to(device)
    running_loss_list=[]
    for epoch in range(num_epochs):
        start = time.time()
        running_loss=0
        steps=0
        trained_images=0
        for inputs, labels in train_loader:
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            trained_images+=len(labels)
            steps+=1
            if steps % print_every == 0:
                print("\rEpoch: [{}/{}]\tLoss: {:.4f}\tImages trained: {} ".format(epoch+1, num_epochs,running_loss/steps,trained_images),end="")
        running_loss_list.append(running_loss/steps)
        if compute_accuracy_at_epoch:
            acc_train,_=compute_accuracy(model,train_loader, classes, device)
            acc_test,_=compute_accuracy(model,test_loader, classes, device)
            print ('\rEpoch [{}/{}], Time: {:.1f}, Average Loss: {:.4f}, Train Accuracy: {:.4f},  Test Accuracy: {:.4f}'.format(epoch+1,num_epochs,(time.time()-start),running_loss/steps,acc_train, acc_test))
        else:
            print ('\rEpoch [{}/{}], Time: {:.1f}, Average Loss: {:.4f}'.format(epoch+1,num_epochs,(time.time()-start),running_loss/steps))
    if plot_training_scores:
        plt.plot(np.arange(num_epochs), np.array(running_loss_list), color='red',linewidth=1.0,)
    plt.show()



def compute_accuracy(model, test_loader, classes, device):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        acc_classes=[]
        for i in range(len(classes)):
            acc_class = 100.0 * n_class_correct[i] / n_class_samples[i]
            acc_classes.append(acc_class)
        model.train()
        return acc,acc_classes
