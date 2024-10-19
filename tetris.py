import random
import cv2
import numpy as np
from PIL import Image
from time import sleep


class Tetris:
    BOARD_WIDTH = 10
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
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (255, 255, 255),
    }

    def __init__(self):
        self.board = np.zeros(
            (Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH), dtype=int)
        self.score = 0
        self.hold = {}
        self.can_hold = True
        self.tetromino_pool = [0, 1, 2, 3, 4, 5, 6]
        random.shuffle(self.tetromino_pool)
        self.next_piece = Tetris.TETROMINOS[self.tetromino_pool.pop()]
        self.curr_piece = self.next_piece
        self.curr_pos = []
        self.game_over = False
        self.next()

    def next(self):
        '''Updates the and curr_piece'''
        if len(self.tetromino_pool) == 0:
            self.tetromino_pool = [0, 1, 2, 3, 4, 5, 6]
            random.shuffle(self.tetromino_pool)
        self.next_piece = Tetris.TETROMINOS[self.tetromino_pool.pop()]
        self.curr_piece = self.next_piece

    def rotate(tetromino):
        '''Rotates the currect tetromino'''
        return [(-y, x) for x, y in tetromino]

    def hold(self):
        '''Holds the current tetromino'''
        if self.can_hold:
            self.hold.append(self.curr_piece)
            self.next_piece = Tetris.TETROMINOS[self.tetromino_pool.pop()]
            self.curr_piece = self.next_piece
            self.can_hold = False

    def play(self, x_pos, render=False, delay=None):
        # Ensures proper starting position of shape and not have "part" of a piece to be on the bottom of the board.
        y_offset = min(y for _, y in self.curr_piece)
        self.curr_pos = [x_pos, -y_offset]

        # Drops the current tetromino while checking for collision
        while not self.check_collision(self.curr_piece, self.curr_pos):
            if render:
                self.render()
                if delay:
                    sleep(delay)
                self.curr_pos[1] += 1
        self.curr_pos[1] -= 1

        self.board = self.add_piece(self.curr_piece, self.curr_pos)

        self.next()

    def check_collision(self, curr_piece, pos):
        '''Checks if the current piece collides with the boundaries or other placed pieces.'''
        for x, y in curr_piece:
            x += pos[0]
            y += pos[1]
            if (x < 0 or x >= Tetris.BOARD_WIDTH or y >= Tetris.BOARD_HEIGHT or (y >= 0 and self.board[y, x] == 1)):
                return True
        return False
    def get_board(self):
        '''Returns the current board'''
        piece = [np.add(x, self.curr_pos) for x in self.curr_piece]
        board = self.board.copy()
        for x, y in piece:
            board[y, x] = 1
        return board
    
    def add_piece(self, piece, pos):
        board = self.board.copy()
        for x, y in piece:
            board[(y + pos[1], x + pos[0])] = 1
        return board
            

    def render(self):
        '''Renders the current board'''
        img = [Tetris.COLORS[p] for row in self.get_board()
               for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT,
                                    Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        img = img.resize(
            (Tetris.BOARD_WIDTH * 100, Tetris.BOARD_HEIGHT * 100), Image.NEAREST)
        # Convert resized image back to a NumPy array for manipulation
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)

    def sc(self, tetro):
        self.curr_piece = Tetris.TETROMINOS[tetro]

# Test Functionally
if __name__ == "__main__":
    game = Tetris()
    game.sc(1)
    game.play(3, True, delay=0.1)
    game.sc(4)
    game.play(3, True, delay=0.1)
    game.play(3, True, delay=0.1)

