import torch
import numpy as np
from torch.utils.data import DataLoader
from model import TruckDroneModel
from dataset import TruckDroneDataset
from utils import setup_logging, load_config
from torch.optim import Adam
from torch.nn import MSELoss
import torch.nn as nn
import logging
import os
import yaml

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(data_path, params):
    """
    Load the dataset and split into training and validation sets.

    Parameters:
    - data_path (str): Path to the data file
    - params (dict): Configuration parameters

    Returns:
    - train_loader (DataLoader): Training data loader
    - val_loader (DataLoader): Validation data loader
    """
    # Load the dataset
    dataset = TruckDroneDataset(data_path, params['data_params'])

    # Split the dataset into training and validation sets
    train_data, val_data = dataset.split(params['split_ratio'])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, params):
    """
    Train the model using the provided training data loader and validate on the validation set.

    Parameters:
    - model (TruckDroneModel): Instance of the TruckDroneModel to train
    - train_loader (DataLoader): Training data loader
    - val_loader (DataLoader): Validation data loader
    - params (dict): Configuration parameters

    Returns:
    - model (TruckDroneModel): Trained model
    """
    # Set the model to training mode
    model.train()

    # Initialize the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    criterion = MSELoss()

    best_val_loss = float('inf')
    for epoch in range(params['num_epochs']):
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        for batch in train_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch['labels'].size(0)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                loss = criterion(outputs, batch['labels'])
                val_loss += loss.item() * batch['labels'].size(0)

        # Calculate average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch+1}/{params['num_epochs']}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(params['model_dir'], 'best_model.pth'))

        # Switch back to training mode
        model.train()

    return model

def main(config_path):
    # Load the configuration file
    with open(config_path, 'r') as file:
        params = load_config(file)

    # Create the output directories if they don't exist
    os.makedirs(params['model_dir'], exist_ok=True)
    os.makedirs(params['log_dir'], exist_ok=True)

    # Initialize the model
    model = TruckDroneModel(params['model_params'])

    # Load the data and split into training and validation sets
    train_loader, val_loader = load_data(params['data_path'], params)

    # Train the model
    model = train_model(model, train_loader, val_loader, params)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(params['model_dir'], 'final_model.pth'))
    logger.info("Training completed. Final model saved.")

if __name__ == '__main__':
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        main(config_path)
    else:
        logger.error("Configuration file not found. Please provide a valid path to the config file.")