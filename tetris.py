import random
import cv2
import numpy as np
from PIL import Image
from time import sleep


class Tetris:
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
    0 : np.array([[1, 1, 1, 1]]), # I
    1 : np.array([[1, 1],         # O
                   [1, 1]]),
    2 : np.array([[0, 1, 0],      # T
                   [1, 1, 1]]), 
    3 : np.array([[0, 1, 1],      # S
                   [1, 1, 0]]),
    4 : np.array([[1, 1, 0],      # Z
                   [0, 1, 1]]),
    5 : np.array([[1, 0, 0],      # J
                   [1, 1, 1]]),
    6 : np.array([[0, 0, 1],      # L
                   [1, 1, 1]])
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
        self.curr_pos = [3, 0]
        self.curr_rotation = 0

        if self.check_collision(self.curr_piece, self.curr_pos):
            self.game_over = True



    def rotate(self, angle):
        '''Change the current rotation'''
        r = self.curr_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def current_rotation(self):
        '''Returns the currect rotation'''
        return self.current_rotation

    def hold(self):
        '''Holds the current tetromino'''
        if self.can_hold:
            self.hold.append(self.curr_piece)
            self.next_piece = Tetris.TETROMINOS[self.tetromino_pool.pop()]
            self.curr_piece = self.next_piece
            self.can_hold = False

    def play(self, x_pos, render=False, delay=None):
        print(self.board)
        # Ensures proper starting position of shape and not have "part" of a piece to be on the bottom of the board.
        self.curr_pos = [x_pos, 0]

        # Drops the current tetromino while checking for collision
        while not self.check_collision(self.curr_piece, self.curr_pos):
            if render:
                self.render()
                if delay:
                    sleep(delay)
                self.curr_pos[1] += 1

        
        self.curr_pos[1] -= 1

        self.board = self.add_piece(self.curr_piece, self.curr_pos)
        self.update_score()
        self.next()

    def check_collision(self, curr_piece, pos):
        '''Checks if the current piece collides with the boundaries or other placed pieces.'''
        piece_height, piece_width = curr_piece.shape
        for idx in range(piece_height):
            for jdx in range(piece_width):
                if curr_piece[idx, jdx] == 1:
                    board_x = pos[0]
                    board_y = pos[1]

                    if board_x < 0 or board_x >= Tetris.BOARD_WIDTH \
                            or board_y < 0 or board_y >= Tetris.BOARD_HEIGHT \
                            or self.board[board_y, board_x] == 1:
                        return True
        return False
        # for x, y in curr_piece:
        #     x += pos[0]
        #     y += pos[1]
        #     # print(f"Checking position: ({x}, {y})")  # Debug output

        #     # print(self.board)

        #     if x < 0 or x >= Tetris.BOARD_WIDTH \
        #             or y < 0 or y >= Tetris.BOARD_HEIGHT \
        #             or self.board[y][x] == 1:
        #         # print(f"Collision detected at: ({x}, {y})")  # Collision detected
        #         return True
        # return False


    def update_score(self):
        '''Updates the score based on the number of full rows'''
        full_rows = self.clean_rows()
        if full_rows:
            self.score += 100 * (2 ** (full_rows - 1))

    def get_board(self):
        '''Returns the current board'''
        piece = [np.add(x, self.curr_pos) for x in self.curr_piece[self.curr_rotation]]
        board = self.board.copy()
        for x, y in piece:
            board[y, x] = 1
        return board

    def add_piece(self, piece, pos):
        # board = self.board.copy()
        # for x, y in piece[self.curr_rotation]:
        #     # Calculate the absolute position on the board
        #     board_x = x + pos[0]
        #     board_y = y + pos[1]

        #     # Check that the position is within bounds of the board
        #     if 0 <= board_x < Tetris.BOARD_WIDTH and 0 <= board_y < Tetris.BOARD_HEIGHT:
        #         board[board_y, board_x] = 1
        # return board
        piece_height, piece_width = piece.shape
        board = self.board.copy()
        for idx in range(piece_height):
            for jdx in range(piece_width):
                if piece[i, j] == 1:
                    board[pos[1] + i, pos[0] + j] = 1
        return board

    def render(self):
        '''Renders the current board'''
        img = np.array([[Tetris.COLORS[cell] for cell in row] for row in self.board], dtype=np.uint8)
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

        for x in range(Tetris.BOARD_HEIGHT):
            for y in range(Tetris.BOARD_WIDTH):
                # Finds the nearest pixel in iterated column and find its height before breaking and iterating to next column.
                if board[y, x] == 1:
                    col_heights[x] = Tetris.BOARD_HEIGHT - y
                    break

        # print(col_heights)
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
                        # print(board[:y, x])
                        col_holes[x] += 1

        # print(col_holes)
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
    while game.game_over != True:
        # game.set_curr()
        game.play(2, True)
    #     game.play(2, True)
    #     game.set_curr(6)
        # game.play(random.randint(1,10), True)
    #     game.set_curr(6)
    #     game.play(6, True)
    #     game.set_curr(6)
    #     game.play(8, True)
    #     game.play(2, True)
    # print("Game Over. Final Score:", game.score)
