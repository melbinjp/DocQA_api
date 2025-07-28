import unittest
from app import embed_model

class TestEmbeddingModel(unittest.TestCase):

    def test_embedding_model(self):
        # Test that the model can be loaded and used to encode a sentence
        sentences = ["This is a test sentence."]
        embeddings = embed_model.encode(sentences)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 384)

if __name__ == '__main__':
    unittest.main()
