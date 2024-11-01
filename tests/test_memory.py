import unittest
import torch
from model.memory import MemoryBank

class TestMemoryBank(unittest.TestCase):
    
    def setUp(self):
        self.memory_size = 3
        self.feature_dim = 5
        self.memory_bank = MemoryBank(self.memory_size, self.feature_dim)

    def test_write_and_read(self):
        input_data = torch.randn(3, self.feature_dim)
        write_weights = torch.softmax(torch.randn(3, self.memory_size), dim=-1)
        self.memory_bank.write(input_data, write_weights)

        query = torch.randn(3, self.feature_dim)
        retrieved_memory, confidence_scores = self.memory_bank.read(query)

        self.assertEqual(retrieved_memory.shape, (3, self.feature_dim))
        self.assertEqual(confidence_scores.shape, (3,))

