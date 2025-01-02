import os
import numpy as np
import random
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# Utility function for generating well logs
def generate_well_log(data):
    """
    Simulates well log data by selecting specific indices in the input.
    
    Args:
        data (np.ndarray): Input seismic data.
    
    Returns:
        np.ndarray: Data with well logs inserted.
    """
    #logs = random.randint(1,8)  # Number of maximum well log points (adjustable)
    well_log = np.zeros_like(data)  # Initialize zero array of the same shape
    known_indices =[0,24,48,69] #np.random.choice(data.shape[2], logs, replace=False)  # Predefined indices for well logs
    well_log[..., known_indices] = data[..., known_indices]  # Copy values at selected indices
    return well_log

# Dataset class for FWI data
class FWIDataset(Dataset):
    """
    Full Waveform Inversion (FWI) Dataset class.
    Handles loading, preprocessing, and transforming seismic and label data.
    
    Args:
        anno (str): Path to annotation file.
        if_test (bool): Indicator for test mode.
        lines (int): Number of lines to load from the annotation file.
        preload (bool): Whether to load the entire dataset into memory.
        sample_ratio (int): Downsampling ratio for seismic data.
        file_size (int): Number of samples in each .npy file.
        transform_data (callable): Transformation for input data.
        transform_label (callable): Transformation for labels.
        conditioning_key (bool): Key for conditional data loading (e.g., seismic, well logs).
    """
    def __init__(self, anno, if_test, lines, preload=True, sample_ratio=1, file_size=500,
                 transform_data=None, transform_label=None, conditioning_key=False):

        self.if_test = if_test
        self.conditioning_key = conditioning_key
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label

        # Read annotation file and load batches
        with open(anno, 'r') as f:
            if lines==0:
                self.batches = f.readlines()
            else:
                self.batches = f.readlines()[:lines]               

        # Preload data and labels if specified
        if preload:
            self.data_list, self.label_list = [], []
            for batch in self.batches:
                data, label = self._load_batch(batch)
                if data is not None:
                    self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    def _load_batch(self, batch):
        """
        Loads a single batch of data and labels.
        Args:
            batch (str): A line from the annotation file.
        
        Returns:
            tuple: Data and label arrays.
        """
        batch_parts = batch.strip().split('\t')
        data_path, label_path = batch_parts[0], batch_parts[1]
        
        data = np.load(data_path).astype('float32') if 'seis' in self.conditioning_key else None
        label = np.load(label_path).astype('float32') if len(batch_parts) > 1 else None

        return data, label

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: A dictionary containing input data, labels, and optional conditioning information.
        """
        batch_idx, sample_idx = divmod(idx, self.file_size)

        # Load data and label
        data = np.copy(self.data_list[batch_idx][sample_idx, :, :, :]) if self.data_list else None
        label = np.copy(self.label_list[batch_idx][sample_idx, :, :, :]) if self.label_list else None

        # Apply transformations if specified
        if self.transform_data and data is not None:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)

        # Prepare output dictionary
        output = {}
        if 'seis' in self.conditioning_key:
            output['seis'] = data
        if 'well' in self.conditioning_key:
            output['well'] = generate_well_log(label)
        if 'back' in self.conditioning_key:
            nx, std = 45,5#random.randint(5,65), random.randint(5,65)  # Gaussian blur parameters
            if nx % 2==0:
                nx=nx+1
                std=std+1
            back = np.zeros_like(label)
            back[0, :, :] = cv2.GaussianBlur(label[0, :, :], (nx, nx), std)
            output['back'] = back

        output['vel'] = label
        return output

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.batches) * self.file_size
