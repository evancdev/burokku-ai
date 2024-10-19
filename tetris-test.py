from tetris import Tetris
import unittest

# # Test 4 Row Clears
# game = Tetris()
# game.set_curr(6)
# game.play(1, True, delay=0.0001)
# game.set_curr(6)
# game.play(3, True, delay=0.0001)
# game.set_curr(6)
# game.play(5, True, delay=0.0001)
# game.set_curr(6)
# game.play(7, True, delay=0.0001)
# game.set_curr(6)
# game.play(1, True, delay=0.0001)
# game.set_curr(6)
# game.play(3, True, delay=0.0001)
# game.set_curr(6)
# game.play(5, True, delay=0.0001)
# game.set_curr(6)
# game.play(7, True, delay=0.0001)
# game.set_curr(5)
# game.play(8, True, delay=0.0001)
# game.set_curr(5)
# game.play(9, True, delay=0.0001)


class TestClass(unittest.TestCase):
    def test_calculate_bumpiness(self):
        self.assertEqual(-1, Tetris.calculate_bumpiness(0))
