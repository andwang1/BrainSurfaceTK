import unittest
from data.get_feature_dict import get_feature_dict

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

SUBJECT = "CC01006XX08_38531"


class TestGetFeature(unittest.TestCase):
    def test_gender(self):
        self.assertEqual(get_feature_dict("gender")[SUBJECT], "Male", "Should be Male")

    def test_birth_age(self):
        self.assertEqual(round(get_feature_dict("birth_age")[SUBJECT], 3), 28.714, "Should be 28.714")

    def test_birth_weight(self):
        self.assertEqual(get_feature_dict("birth_weight")[SUBJECT], 1.03, "Should be 1.03")

    def test_scan_age(self):
        self.assertEqual(round(get_feature_dict("scan_age")[SUBJECT], 3), 30.571, "Should be 1.03")

    def test_scan_number(self):
        self.assertEqual(get_feature_dict("scan_number")[SUBJECT], 1, "Should be 1")


if __name__ == '__main__':
    unittest.main()
