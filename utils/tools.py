#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for experiment preparation and execution.
"""

import os
import time
from datetime import datetime


def create_experiment_dir(base_dir='results'):
    """
    Create a directory for the current experiment.
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        str: Path to the experiment directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir)
    
    return experiment_dir