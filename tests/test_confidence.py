import unittest
import torch
from model.confidence import ConfidenceScoring

class TestConfidenceScoring(unittest.TestCase):
    
    def setUp(self):
        self.feature_dim = 10
        self.confidence_scoring = ConfidenceScoring(self.feature_dim)

    def test_calculate_confidence(self):
        query = torch.randn(3, self.feature_dim)
        retrieved = torch.randn(3, self.feature_dim)
        confidence_scores = self.confidence_scoring.calculate_confidence(query, retrieved)
        
        self.assertEqual(confidence_scores.shape, (3,))

