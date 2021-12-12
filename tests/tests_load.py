import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/pynirs")

FILE_0 = os.path.join(os.path.dirname(__file__), 'test_set0.csv')

from nirs_set import NIRS_Set as NSet

class TestLoadMethods(unittest.TestCase):
    
    def test_singleFile(self):
        n = NSet("test")
        n.load(FILE_0, y_cols = 0)
        self.assertEqual(n.get_raw_x().shape, (3,62))
        self.assertEqual(n.get_raw_y().shape, (3,1))
        
        n.load(FILE_0, y_cols = [0,1])
        self.assertEqual(n.get_raw_x().shape, (3,61))
        self.assertEqual(n.get_raw_y().shape, (3,2))
        



if __name__ == '__main__':
    runittest.main()