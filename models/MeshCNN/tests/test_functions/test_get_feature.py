import sys
from data.get_feature_dict import get_feature_dict

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

CC01006XX08_38531_features = ["Male", 28.714285714285715, 1.03, 30.571428571429, 1]
features = ["gender", "birth_age", "birth_weight", "scan_age", "scan_number"]

def test():
    test_pass = True
    for i, feature in enumerate(features):
        actual_val = CC01006XX08_38531_features[i]
        try:
            test_val = get_feature_dict(feature)["CC01006XX08_38531"]
        except:
            test_pass = False
            print(f"Failed to get function value for {feature}.")
            continue
        try:
            assert test_val == actual_val
        except:
            test_pass = False
            print(f"Failed to get feature {feature}. Expected {actual_val} of type {type(actual_val)}, got {test_val} of type {type(test_val)}")
    return test_pass

if __name__ == "__main__":
    test_passed = test()
    print("get_feature_dict test passed = ", test_passed)
