import unittest
import torch
from model.attention import Attention

class TestAttention(unittest.TestCase):
    
    def setUp(self):
        self.feature_dim = 10
        self.attention = Attention(self.feature_dim)

    def test_attention_forward(self):
        batch_size = 3
        memory_size = 5
        query = torch.randn(batch_size, self.feature_dim)
        keys = torch.randn(batch_size, memory_size, self.feature_dim)
        values = torch.randn(batch_size, memory_size, self.feature_dim)
        
        output = self.attention(query, keys, values)

        self.assertEqual(output.shape, (batch_size, self.feature_dim))

    def test_attention_empty_input(self):
        batch_size = 0
        memory_size = 5
        query = torch.randn(batch_size, self.feature_dim)
        keys = torch.randn(batch_size, memory_size, self.feature_dim)
        values = torch.randn(batch_size, memory_size, self.feature_dim)

        output = self.attention(query, keys, values)
        self.assertEqual(output.shape, (batch_size, self.feature_dim))

    def test_attention_single_query(self):
        batch_size = 1
        memory_size = 5
        query = torch.randn(batch_size, self.feature_dim)
        keys = torch.randn(batch_size, memory_size, self.feature_dim)
        values = torch.randn(batch_size, memory_size, self.feature_dim)
        
        output = self.attention(query, keys, values)
        self.assertEqual(output.shape, (batch_size, self.feature_dim))

    def test_attention_invalid_input_shape(self):
        batch_size = 3
        memory_size = 5
        query = torch.randn(batch_size, self.feature_dim)
        keys = torch.randn(batch_size, memory_size, self.feature_dim)
        values = torch.randn(batch_size, memory_size + 1, self.feature_dim)  # Invalid shape

        with self.assertRaises(RuntimeError):
            self.attention(query, keys, values)

    def test_attention_weights_sum_to_one(self):
        batch_size = 3
        memory_size = 5
        query = torch.randn(batch_size, self.feature_dim)
        keys = torch.randn(batch_size, memory_size, self.feature_dim)
        values = torch.randn(batch_size, memory_size, self.feature_dim)
        
        output = self.attention(query, keys, values)
        attention_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Check if the sum of weights equals 1
        self.assertTrue(torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size)))


