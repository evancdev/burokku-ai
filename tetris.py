import random
import cv2 as cv
import numpy as np

class Tetris:
    BOARD_WIDHT = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # Square
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        }
    }

    COLORS = {
    }


    def __init__(self):
        self.start()


    def start(self):
        '''Starts the game'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.score = 0
        self.tetromino_pool = self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.tetromino_pool)
        self.next_piece = self.pool.pop()
        self.next_move()

    def next_move(self):
        pass
  
    # 
    def get_score(self):
      """Summary or Description of the Function
      """
      
      return self.score
    
  
    def calculate_bumpiness(self):
        """Returns the bumpiness of a grid. The bumpiness of two columns is defined to be the variation between adjacent column heights."""
        
        bumpiness = 0
        
        # Initiaize empty array that calculates the height of each column of the grid
        col_height = []

        for col in range(Tetris.BOARD_WIDHT):
            col_height[col] = 0

        for idx in range(Tetris.BOARD_HEIGHT):
            for jdx in range(Tetris.BOARD_WIDHT):
                continue

        pass
