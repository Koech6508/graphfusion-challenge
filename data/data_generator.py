import torch
import numpy as np

def generate_sequence_data(seq_length: int, feature_dim: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequential data for testing memory retention.
    
    Args:
        seq_length: Length of sequences
        feature_dim: Dimension of features
        batch_size: Number of sequences
        
    Returns:
        tuple: (input_sequences, target_sequences)
    """
    # Generate random sequential data as input
    input_sequences = torch.randn(batch_size, seq_length, feature_dim)
    
    # Target sequences could be a shifted version of input for sequence prediction tasks
    target_sequences = input_sequences.roll(shifts=-1, dims=1)  # Shifted by 1 for prediction
    
    return input_sequences, target_sequences
