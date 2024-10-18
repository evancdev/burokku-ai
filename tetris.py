import random
import numpy as np


class Tetris:
    BOARD_WIDHT = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: [(0, 0), (-1, 0), (1, 0), (0, -1)],  # T
        1: [(0, 0), (-1, 0), (0, -1), (0, -2)],  # J
        2: [(0, 0), (1, 0), (0, -1), (0, -2)],  # L
        3: [(0, 0), (-1, 0), (0, -1), (1, -1)],  # Z
        4: [(0, 0), (-1, -1), (0, -1), (1, 0)],  # S
        5: [(0, 0), (0, -1), (0, -2), (0, -3)],  # I
        6: [(0, 0), (0, -1), (-1, 0), (-1, -1)],  # O
    }

    COLORS = {
    }

    def __init__(self):
        self.start()

    def start(self):
        '''Starts the game'''
        self.board = np.zeros(
            shape=(Tetris.BOARD_WIDTH, Tetris.BOARD_HEIGH), dtype=np.float)
        self.score = 0
        self.hold = {}
        self.can_hold = True
        self.tetromino_pool = self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.tetromino_pool)
        self.next_piece = self.tetromino_pool.pop()
        self.curr_piece = self.next_piece
        self.game_over = False
        self.next_move()

    def get_score(self):
        '''Returns the current score'''
        return self.score

    def next_move(self):
        '''Player plays the next move'''
        if len(self.tetromino_pool) == 0:
            self.tetromino_pool = self.bag = list(
                range(len(Tetris.TETROMINOS)))
            random.shuffle(self.tetromino_pool)
        self.next_piece = self.tetromino_pool.pop()

        self.game_over = self.check_game()

    def rotate(tetromino):
        '''Rotates the currect tetromino'''
        return [(-y, x) for x, y in tetromino]

    def hold(self):
        '''Holds the current tetromino'''
        if self.can_hold:
            self.hold.append(self.next_piece)
            self.next_piece = self.pool.pop()
            self.can_hold = False

    def check_game(self):
        '''Checks if the game is over. Returns T when the game is over.'''

    def play(self, render=False):
