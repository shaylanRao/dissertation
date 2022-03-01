import unittest
from hashtagSearch import *


class TestSentiAnalyser(unittest.TestCase):

    def test_sample(self):
        result = get_trackid_from_urls(10, 5)
        self.assertEqual(result, 15)

    def test_get_text_senti(self):
        pass


if __name__ == '__main__':
    unittest.main()

# Use below to run code in terminal
# python -m unittest test_sentiment_analyser.py
