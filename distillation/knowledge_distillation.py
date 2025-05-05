#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of knowledge distillation techniques for model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DistillationLoss(nn.Module):
    """
    Loss function for knowledge distillation.
    Combines cross-entropy loss with KL divergence between teacher and student outputs.
    """
    def __init__(self, alpha=0.5, temperature=4.0):
        """
        Initialize the distillation loss.
        
        Args:
            alpha (float): Weight for the distillation loss (0.0-1.0)
            temperature (float): Temperature for softening the teacher outputs
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Compute the distillation loss.
        
        Args:
            student_outputs (torch.Tensor): Outputs from the student model
            teacher_outputs (torch.Tensor): Outputs from the teacher model
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Hard loss (cross-entropy with ground truth)
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Soft loss (KL divergence between teacher and student)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss

def train_with_distillation(teacher_model, student_model, train_loader, val_loader, device, 
                           alpha=0.5, temperature=4.0, epochs=10, lr=0.001):
    """
    Train a student model using knowledge distillation from a teacher model.
    
    Args:
        teacher_model (nn.Module): Teacher model
        student_model (nn.Module): Student model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use
        alpha (float): Weight for the distillation loss (0.0-1.0)
        temperature (float): Temperature for softening the teacher outputs
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        nn.Module: Trained student model
    """
    # Create a copy of the student model to avoid modifying the original
    student = copy.deepcopy(student_model).to(device)
    
    # Set teacher model to evaluation mode
    teacher = copy.deepcopy(teacher_model).to(device)
    teacher.eval()
    
    # Define loss function and optimizer
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            
            # Compute loss
            loss = criterion(student_outputs, teacher_outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        
        # Validation
        student.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                student_outputs = student(inputs)
                teacher_outputs = teacher(inputs)
                
                # Compute loss
                loss = criterion(student_outputs, teacher_outputs, targets)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
    
    return student

class FeatureDistillationLoss(nn.Module):
    """
    Loss function for feature-based knowledge distillation.
    Combines cross-entropy loss with MSE between teacher and student features.
    """
    def __init__(self, alpha=0.5):
        """
        Initialize the feature distillation loss.
        
        Args:
            alpha (float): Weight for the feature distillation loss (0.0-1.0)
        """
        super(FeatureDistillationLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_outputs, teacher_outputs, student_features, teacher_features, targets):
        """
        Compute the feature distillation loss.
        
        Args:
            student_outputs (torch.Tensor): Outputs from the student model
            teacher_outputs (torch.Tensor): Outputs from the teacher model
            student_features (list): Features from the student model
            teacher_features (list): Features from the teacher model
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Hard loss (cross-entropy with ground truth)
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Feature loss (MSE between teacher and student features)
        feature_loss = 0.0
        for sf, tf in zip(student_features, teacher_features):
            # Adapt feature dimensions if needed
            if sf.size() != tf.size():
                # Use adaptive pooling to match spatial dimensions
                sf = F.adaptive_avg_pool2d(sf, tf.size()[2:])
            feature_loss += self.mse_loss(sf, tf)
        
        # Combined loss
        return (1 - self.alpha) * hard_loss + self.alpha * feature_loss

class RelationalDistillationLoss(nn.Module):
    """
    Loss function for relational knowledge distillation.
    Preserves the relationships between samples in a batch.
    """
    def __init__(self, alpha=0.5, temperature=4.0):
        """
        Initialize the relational distillation loss.
        
        Args:
            alpha (float): Weight for the relational distillation loss (0.0-1.0)
            temperature (float): Temperature for softening the relations
        """
        super(RelationalDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Compute the relational distillation loss.
        
        Args:
            student_outputs (torch.Tensor): Outputs from the student model
            teacher_outputs (torch.Tensor): Outputs from the teacher model
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Hard loss (cross-entropy with ground truth)
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Compute relations (cosine similarity)
        student_relations = self._get_relations(student_outputs)
        teacher_relations = self._get_relations(teacher_outputs)
        
        # Relational loss (MSE between teacher and student relations)
        relational_loss = F.mse_loss(student_relations, teacher_relations)
        
        # Combined loss
        return (1 - self.alpha) * hard_loss + self.alpha * relational_loss
    
    def _get_relations(self, outputs):
        """
        Compute the relations between samples in a batch.
        
        Args:
            outputs (torch.Tensor): Model outputs
            
        Returns:
            torch.Tensor: Relation matrix
        """
        # Normalize outputs
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        
        # Compute cosine similarity
        relations = torch.mm(outputs_norm, outputs_norm.t())
        
        # Apply temperature
        relations = relations / self.temperature
        
        return relations

def train_with_feature_distillation(teacher_model, student_model, train_loader, val_loader, device,
                                   teacher_features_idx, student_features_idx,
                                   alpha=0.5, epochs=10, lr=0.001):
    """
    Train a student model using feature-based knowledge distillation from a teacher model.
    
    Args:
        teacher_model (nn.Module): Teacher model
        student_model (nn.Module): Student model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use
        teacher_features_idx (list): Indices of teacher model layers to extract features from
        student_features_idx (list): Indices of student model layers to extract features from
        alpha (float): Weight for the feature distillation loss (0.0-1.0)
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        nn.Module: Trained student model
    """
    # Create a copy of the student model to avoid modifying the original
    student = copy.deepcopy(student_model).to(device)
    
    # Set teacher model to evaluation mode
    teacher = copy.deepcopy(teacher_model).to(device)
    teacher.eval()
    
    # Define hooks to extract features
    teacher_features = []
    student_features = []
    
    def get_teacher_hook(idx):
        def hook(module, input, output):
            if len(teacher_features) <= idx:
                teacher_features.append(output)
            else:
                teacher_features[idx] = output
        return hook
    
    def get_student_hook(idx):
        def hook(module, input, output):
            if len(student_features) <= idx:
                student_features.append(output)
            else:
                student_features[idx] = output
        return hook
    
    # Register hooks
    teacher_hooks = []
    student_hooks = []
    
    for i, idx in enumerate(teacher_features_idx):
        teacher_module = list(teacher.modules())[idx]
        hook = teacher_module.register_forward_hook(get_teacher_hook(i))
        teacher_hooks.append(hook)
    
    for i, idx in enumerate(student_features_idx):
        student_module = list(student.modules())[idx]
        hook = student_module.register_forward_hook(get_student_hook(i))
        student_hooks.append(hook)
    
    # Define loss function and optimizer
    criterion = FeatureDistillationLoss(alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Clear features
            teacher_features.clear()
            student_features.clear()
            
            # Forward pass
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            
            # Compute loss
            loss = criterion(student_outputs, teacher_outputs, student_features, teacher_features, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        
        # Validation
        student.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Clear features
                teacher_features.clear()
                student_features.clear()
                
                # Forward pass
                teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)
                
                # Compute loss
                loss = criterion(student_outputs, teacher_outputs, student_features, teacher_features, targets)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
    
    # Remove hooks
    for hook in teacher_hooks:
        hook.remove()
    for hook in student_hooks:
        hook.remove()
    
    return student
