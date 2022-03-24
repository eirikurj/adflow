import unittest
import numpy
import os
import sys
baseDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(baseDir,'../../'))
from adflow import ADFLOW


class BasicTests(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        gridFile = '../input_files/mdo_tutorial_euler.cgns'
        options = {'gridfile': gridFile}
        self.CFDSolver = ADFLOW(options=options)

    def test_import(self):
        gridFile = "input_files/mdo_tutorial_euler.cgns"
        options = {"gridfile": os.path.join(baseDir, "../../", gridFile)}
        ADFLOW(options=options, debug=False)


if __name__ == "__main__":
    unittest.main()
