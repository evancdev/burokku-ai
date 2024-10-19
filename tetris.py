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

    def rotate(self):
        '''Rotates the currect tetromino'''
        self.curr_piece = [(-y, x) for x, y in self.curr_piece]

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
                # if delay:
                # sleep(delay)
                self.curr_pos[1] += 1
        self.curr_pos[1] -= 1

        self.add_piece(self.curr_piece, self.curr_pos)
        self.update_score()
        self.next()

    def check_collision(self, curr_piece, pos):
        '''Checks if the current piece collides with the boundaries or other placed pieces.'''
        for x, y in curr_piece:
            x += pos[0]
            y += pos[1]
            if (x < 0 or x >= Tetris.BOARD_WIDTH or y >= Tetris.BOARD_HEIGHT or (y >= 0 and self.board[y, x] == 1)):
                return True
        return False

    def update_score(self):
        '''Updates the score based on the number of full rows'''
        full_rows = self.clean_rows()
        if full_rows:
            self.score += 100 * (2 ** (full_rows - 1))

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
            # Calculate the absolute position on the board
            board_x = x + pos[0]
            board_y = y + pos[1]

            # Check that the position is within bounds of the board
            if 0 <= board_x < Tetris.BOARD_WIDTH and 0 <= board_y < Tetris.BOARD_HEIGHT:
                board[board_y, board_x] = 1
        self.board = board

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

    def set_curr(self, tetromino):
        '''Sets the current piece to tetromino. Used for debugging'''
        self.curr_piece = Tetris.TETROMINOS[tetromino]

    def set_board(self, nboard):
        '''Sets the current board to nboard. nboard has to be a np array'''
        self.board = nboard

    def print_score(self):
        '''Prints the score'''
        print(self.score)

    def print_board(self):
        '''Prints the board'''
        print(self.board)

    # STATISTICS #

    def calculate_bumpiness(self, board):
        '''
        Given a board, calculate the difference of heights between two adjacent columns.
        An undesirable board is one where there exists deep "wells"
        '''
        bumpiness = 0
        col_heights = [0 for _ in range(Tetris.BOARD_WIDTH)]

        for x in range(Tetris.BOARD_WIDTH):
            for y in range(Tetris.BOARD_HEIGHT):
                # Finds the nearest pixel in iterated column and find its height before breaking and iterating to next column.
                if board[y, x] == 1:
                    col_heights[x] = Tetris.BOARD_HEIGHT - y
                    break

        print(col_heights)
        for idx in range(1, len(col_heights)):
            bumpiness += abs(col_heights[idx]-col_heights[idx-1])

        return bumpiness

    def calculate_holes(self, board):
        '''
        Given a board, calculate the number of holes that exist within the board.
        A "hole" is defined when there exists an empty pixel and there exists a placed pixel above it in the same column.
        '''

        holes = 0
        col_holes = [0 for _ in range(Tetris.BOARD_WIDTH)]

        for x in range(Tetris.BOARD_WIDTH):
            for y in range(Tetris.BOARD_HEIGHT):
                if board[y, x] == 0:
                    if 1 in board[:y, x]:
                        print(board[:y, x])
                        col_holes[x] += 1

        print(col_holes)
        holes = sum(col_holes)
        return holes

    def clean_rows(self):
        '''Updates the board based on the number of full rows. Returns the number of full rows.'''
        # Finds all the rows that are full
        full_rows = [index for index, row in enumerate(
            self.board) if np.all(row)]

        # Remove full rows and shift the rest down
        for row_index in full_rows:
            self.board = np.delete(self.board, row_index, axis=0)
            new_row = np.zeros((1, Tetris.BOARD_WIDTH), dtype=int)
            self.board = np.vstack([new_row, self.board])
        return len(full_rows)

    def aggregated_height(self):
        ''' Computes the aggregated height and returns it '''
        heights = []
        for col in range(Tetris.BOARD_WIDTH):
            column_tiles = self.board[:, col]
            non_zero = np.where(column_tiles > 0)[0]
            if len(non_zero) > 0:
                height = Tetris.BOARD_HEIGHT - non_zero[0]
            else:
                height = 0
            heights.append(height)

        return sum(heights)


# Test Functionally
if __name__ == "__main__":
    game = Tetris()
