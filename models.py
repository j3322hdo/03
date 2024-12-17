from torch import nn
import torch
import models
import time

device = "cpu" if torch.cuda.is_available() else "cpu"

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatton = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatton(x)
        logits = self.network(x)
        return logits

def test_accuracy(model,dateloader,device="cpu"):
    n_corrests = 0
    
    model.to(device)
    for image_batch, label_batch in dateloader:
        image_batch.to(device)
        label_batch.to(device)
        
        with torch.no_grad():
            logits_batch =model(image_batch)
        
        predict_batch = logits_batch.argmax(dim=1)
        n_corrests += (label_batch == predict_batch).sum().item()
    
    accuracy = n_corrests / len(dateloader.dataset)
    
    return accuracy


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    model.to(device)
    for image_batch, label_batch in dataloader:
        image_batch.to(device)
        label_batch.to(device)
        
        logits_batch = model(image_batch)
        
        loss = loss_fn(logits_batch, label_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()  

def test(model, dataloader, loss_fn):
    model.to(device)
    model.eval()
    loss_total = 0.0
    
    for image_batch, label_batch in dataloader:
        image_batch.to(device)
        label_batch.to(device)
        with torch.no.grad():
            logits_batch = model(image_batch)
            
        loss = loss_fn(logits_batch, label_batch)
        loss_total += loss.item()
        
    return loss_total / len(dataloader)

