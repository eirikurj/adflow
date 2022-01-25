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
        "Tests if mesh was read properly"
        nStates = self.CFDSolver.getStateSize()
        assert(nStates == 72576)

if __name__ == '__main__':
    unittest.main()
