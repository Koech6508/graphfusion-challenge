import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention scores and output.

        Args:
            query: Input query tensor of shape (batch_size, feature_dim)
            keys: Input keys tensor of shape (batch_size, memory_size, feature_dim)
            values: Input values tensor of shape (batch_size, memory_size, feature_dim)

        Returns:
            Output tensor after applying attention.
        """
        # Transform the query, keys, and values
        query_transformed = self.query_layer(query)
        keys_transformed = self.key_layer(keys)
        values_transformed = self.value_layer(values)

        # Compute attention scores
        
        
        # Transform the query, keys, and values
        query_transformed = self.query_layer(query).unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)
        keys_transformed = self.key_layer(keys)  # Shape: (batch_size, memory_size, feature_dim)
        values_transformed = self.value_layer(values)  # Shape: (batch_size, memory_size, feature_dim)

    # Compute attention scores
        attention_scores = torch.bmm(query_transformed, keys_transformed.transpose(1, 2))  # Shape: (batch_size, 1, memory_size)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Normalize scores

    # Compute the output as a weighted sum of values
        output = torch.bmm(attention_weights, values_transformed)  # Shape: (batch_size, 1, feature_dim)
        output = output.squeeze(1)  # Remove the second dimension

    
        return output
