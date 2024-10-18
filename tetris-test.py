from tetris import Tetris
import unittest

class TestClass(unittest.TestCase):
  def test_calculate_bumpiness(self):
    self.assertEqual(-1, Tetris.calculate_bumpiness(0))



