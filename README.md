# GraphFusion ML Engineering Challenge
## Neural Memory Network Implementation

### Overview
Build a simple neural memory network that demonstrates your understanding of ML fundamentals and memory systems. This challenge is designed to be completed in 4-8 hours.

### Challenge Description
Create a memory-augmented neural network that can:
1. Store and retrieve information from a memory bank
2. Implement attention-based memory access
3. Calculate confidence scores for retrievals
4. Demonstrate learning over sequential data

### Repository Structure
```
graphfusion-challenge/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── data_generator.py
├── model/
│   ├── __init__.py
│   ├── memory.py
│   ├── attention.py
│   └── confidence.py
├── tests/
│   ├── __init__.py
│   ├── test_memory.py
│   ├── test_attention.py
│   └── test_confidence.py
├── notebooks/
│   └── demo.ipynb
└── evaluation/
    └── metrics.py
```

### Starter Code

`model/memory.py`:
```python
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
        
    def write(self, input_data: torch.Tensor, write_weights: torch.Tensor) -> None:
        """
        Write data to memory using attention weights.
        
        Args:
            input_data: Data to write (batch_size, feature_dim)
            write_weights: Where to write (batch_size, memory_size)
        """
        # TODO: Implement memory writing mechanism
        pass
        
    def read(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using attention mechanism.
        
        Args:
            query: Query vector (batch_size, query_dim)
            
        Returns:
            tuple: (retrieved_memory, confidence_scores)
        """
        # TODO: Implement memory reading mechanism
        pass

class ConfidenceScoring(nn.Module):
    def __init__(self, feature_dim: int):
        """
        Initialize confidence scoring mechanism.
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        super().__init__()
        # TODO: Implement confidence scoring initialization
        
    def calculate_confidence(self, query: torch.Tensor, 
                           retrieved: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence score for retrieved memories.
        
        Args:
            query: Original query vector
            retrieved: Retrieved memory vector
            
        Returns:
            Confidence score between 0 and 1
        """
        # TODO: Implement confidence calculation
        pass

```

`data/data_generator.py`:
```python
import torch
import numpy as np

def generate_sequence_data(seq_length: int, 
                         feature_dim: int, 
                         batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequential data for testing memory retention.
    
    Args:
        seq_length: Length of sequences
        feature_dim: Dimension of features
        batch_size: Number of sequences
        
    Returns:
        tuple: (input_sequences, target_sequences)
    """
    # TODO: Implement data generation
    pass

def generate_memory_test_data(num_items: int, 
                            feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate test data for memory storage and retrieval.
    
    Args:
        num_items: Number of items to generate
        feature_dim: Dimension of features
        
    Returns:
        tuple: (store_data, query_data)
    """
    # TODO: Implement test data generation
    pass
```

### Requirements
1. Complete the TODO sections in the starter code
2. Implement tests demonstrating functionality
3. Create a notebook showing example usage
4. Document your design decisions
5. Include performance metrics

### Evaluation Criteria
1. **Code Quality (30%)**
   - Clean, readable code
   - Good documentation
   - Proper error handling
   - Effective testing

2. **Technical Implementation (40%)**
   - Memory mechanism effectiveness
   - Attention implementation
   - Confidence scoring accuracy
   - Performance optimization

3. **Innovation (20%)**
   - Novel approaches
   - Creative solutions
   - Additional features

4. **Documentation (10%)**
   - Clear explanations
   - Design justification
   - Usage examples

### Submission Instructions
1. Fork this repository
2. Complete the implementation
3. Add documentation
4. Create a pull request
5. Include a brief write-up of your approach

### Timeline
- Please complete within one week
- Time spent should be 4-8 hours
- Quality > Speed

### Evaluation Metrics
Your implementation will be tested on:
1. Memory retention accuracy
2. Retrieval precision
3. Confidence score correlation
4. Training stability
5. Inference speed

### Next Steps
After submission:
1. Code review discussion
2. Technical deep dive
3. Architecture discussion
4. System design conversation

Questions? Contact: [careers@GraphFusion.onmicrosoft.com]
