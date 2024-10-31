import unittest
import tempfile
import os, sys
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from unittest import mock

# Add the scripts directory directly to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))


from scripts.utils import pickle_object, load_pickle, use_label_encoder, load_mlflow_model 

class TestFunctions(unittest.TestCase):

    def test_pickle_object_and_load_pickle(self):
        # Create a temporary file path
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name

        # Test object
        test_data = {"key": "value", "number": 42}
        
        # Test pickle_object
        pickle_object(file_path, test_data)
        
        # Test load_pickle
        loaded_data = load_pickle(file_path)
        
        # Assert that the data saved and loaded are the same
        self.assertEqual(test_data, loaded_data)

        # Clean up the temporary file
        os.remove(file_path)

    def test_use_label_encoder(self):
        # Sample data and label encoder
        data = pd.Series(["apple", "banana", "orange", "apple", "unknown"])
        encoder = LabelEncoder()
        encoder.fit(["apple", "banana", "orange"])
        
        # Expected results (unknown should map to -1)
        expected_results = pd.Series([0, 1, 2, 0, -1])
        
        # Test use_label_encoder
        encoded_data = use_label_encoder(data, encoder)
        
        # Assert that the encoded data matches the expected results
        pd.testing.assert_series_equal(encoded_data, expected_results)

if __name__ == '__main__':
    unittest.main()
