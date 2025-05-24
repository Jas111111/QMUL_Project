#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of train process for baseline.
"""

import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
# from torch.amp import autocast

def evaluate_model(model, test_loader, device=torch.device("cpu"), criterion=None):
    inference_times = []
    model.eval()
    model.to(device)
    running_loss = 0
    running_corrects = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        start_time = time.time()
        outputs = model(inputs)
        end_time = time.time()
        _, preds = torch.max(outputs, 1)
        inference_times.append(end_time - start_time)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    avg_inference_time = sum(inference_times) / max(len(inference_times), 1)
    
    return {
        'accuracy': eval_accuracy,
        'loss': eval_loss,
        'inference_time': avg_inference_time,
        'inference_times': inference_times
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc.item()

def train_baseline_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.56), int(epochs * 0.78)], gamma=0.1, last_epoch=-1)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        model.eval()
        val_metric = evaluate_model(model, val_loader, device=device, criterion=criterion)
        val_loss = val_metric["loss"]
        val_acc = val_metric["accuracy"]

        scheduler.step()
        print("\t last learning rate:", scheduler.get_last_lr())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_acc*100:.2f}%")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model
