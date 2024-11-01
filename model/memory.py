import torch
import torch.nn as nn

class MemoryBank(nn.Module):
    def __init__(self, memory_size: int, feature_dim: int):
        """
        Initialize a differentiable memory bank.
        
        Args:
            memory_size: Number of memory slots
            feature_dim: Dimension of each memory slot
        """
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        # Initialize memory bank
        # TODO: Implement memory initialization
        self.memory_bank = nn.Parameter(torch.randn(memory_size, feature_dim) * 0.01)
        
    def write(self, input_data: torch.Tensor, write_weights: torch.Tensor) -> None:
        """
        Write data to memory using attention weights.
        
        Args:
            input_data: Data to write (batch_size, feature_dim)
            write_weights: Where to write (batch_size, memory_size)
        """
        # TODO: Implement memory writing mechanism
           # Ensure weighted_data has the correct dimensions
        weighted_data = torch.matmul(write_weights.T, input_data)  # Shape: (memory_size, feature_dim)
        self.memory_bank.data = self.memory_bank.data + weighted_data
        
        
        
    def read(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using attention mechanism.
        
        Args:
            query: Query vector (batch_size, query_dim)
            
        Returns:
            tuple: (retrieved_memory, confidence_scores)
        """
        # TODO: Implement memory reading mechanism
        
        attention_scores = torch.matmul(query, self.memory_bank.T)  # (batch_size, memory_size)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Normalize scores
        retrieved_memory = torch.matmul(attention_weights, self.memory_bank)  # Weighted sum
        confidence_scores = attention_weights.max(dim=-1).values  # Highest attention score per batch
        
        return retrieved_memory, confidence_scores
        

