#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for logging.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logger(log_file=None):
    """
    Set up a logger for the project.
    
    Args:
        log_file (str): Path to log file. If None, logs will only be printed to console.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('dnn_optimization')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
