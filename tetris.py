import random
import cv2
import numpy as np
from PIL import Image


class Tetris:
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: np.array([[1, 1, 1, 1]]),  # I
        1: np.array([[1, 1],          # O
                     [1, 1]]),
        2: np.array([[0, 1, 0],       # T
                     [1, 1, 1]]),
        3: np.array([[0, 1, 1],       # S
                     [1, 1, 0]]),
        4: np.array([[1, 1, 0],       # Z
                     [0, 1, 1]]),
        5: np.array([[1, 0, 0],       # J
                     [1, 1, 1]]),
        6: np.array([[0, 0, 1],       # L
                     [1, 1, 1]])
    }

    COLORS = {
        0: (0, 0, 0),       # Empty (black)
        1: (255, 255, 255),  # Filled (white)
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(
            (Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH), dtype=int)

        # self.board = np.array([
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        #     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        #     [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        #     [1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        #     [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #     [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        # ])

        self.score = 0
        self.lines_cleared = 0
        self.hold_piece = None
        self.can_hold = True

        self.tetromino_pool = list(Tetris.TETROMINOS.keys())
        random.shuffle(self.tetromino_pool)
        self.next_piece = Tetris.TETROMINOS[self.tetromino_pool.pop()]
        self.curr_piece = self.next_piece

        self.curr_pos = [3, 0]  # Start near the top center

        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        '''Spawns the next piece'''

        if len(self.tetromino_pool) == 0:
            self.tetromino_pool = list(Tetris.TETROMINOS.keys())
            random.shuffle(self.tetromino_pool)

        self.curr_piece = self.next_piece
        self.next_piece = Tetris.TETROMINOS[self.tetromino_pool.pop()]

        self.curr_pos = [3, 0]

    def rotate_piece(self, rotation):
        '''Rotates the current piece 90 degrees counterclockwise'''

        rotated_piece = np.rot90(self.curr_piece, k=rotation)

        if not self.check_collision(rotated_piece, self.curr_pos):
            self.curr_piece = rotated_piece

    def play(self, x_pos, render=False, delay=None):
        '''Handles the game logic for placing pieces'''

        self.curr_pos = [x_pos, 0]

        # Check if shape is valid to place based on its width and if it may cross boundaries
        if x_pos < 0 or x_pos + self.curr_piece.shape[1] > Tetris.BOARD_WIDTH:
            return

        # Drop the Tetromino gradually until it collides
        while not self.check_collision(self.curr_piece, self.curr_pos):
            if render:
                self.render()  # Render the game to show the piece moving
                if delay:
                    sleep(delay)  # Add a delay to simulate falling

            # Increment the Y position to simulate falling
            self.curr_pos[1] += 1

        # Once a collision is detected, move the piece back up by 1 row
        self.curr_pos[1] -= 1

        # Place the piece on the board
        self.board = self.add_piece(self.curr_piece, self.curr_pos)

        # Update the score based on cleared rows
        self.update_score()

        # Spawn the next piece
        self.spawn_piece()

        if render:
            self.render()

        return self.score, self.game_over

    def check_collision(self, piece, pos):
        '''Checks for collisions with the board boundaries and other blocks'''

        piece_height, piece_width = piece.shape

        # Chcek if piece fits horizontally
        if pos[0] < 0 or pos[0] + piece_width > Tetris.BOARD_WIDTH:
            # print(pos[0])
            # print(piece_width)
            return True

        for i in range(piece_height):
            for j in range(piece_width):
                if piece[i, j] == 1:
                    board_x = pos[0] + j
                    board_y = pos[1] + i

                    # Check if piece fits on the floor and not outside
                    if board_y >= Tetris.BOARD_HEIGHT:
                        return True

                    # Check nearest tile collided
                    if board_y >= 0 and self.board[board_y, board_x] == 1:
                        return True  # Collied with a placed tile

        return False

    def add_piece(self, piece, pos):
        '''Adds the piece to the board at the given position'''

        try:

            piece_height, piece_width = piece.shape
            board = self.board.copy()
            for i in range(piece_height):
                for j in range(piece_width):
                    if piece[i, j] == 1:
                        board[pos[1] + i, pos[0] + j] = 1

            # Check game over condition
            if self.check_collision(self.curr_piece, self.curr_pos):
                self.game_over = True

            return board

        except:
            print(pos)
            print(self.board)
            print(self.curr_piece)

    def update_score(self):
        '''Updates score after clearing full rows'''

        full_rows = self.clean_rows()
        if full_rows:
            self.score += 1 * (2 ** (full_rows - 1))

    def clean_rows(self):
        '''Clears full rows and shifts the rest down'''

        full_rows = [i for i, row in enumerate(self.board) if np.all(row)]
        for row_index in full_rows:
            self.board = np.delete(self.board, row_index, axis=0)
            new_row = np.zeros((1, Tetris.BOARD_WIDTH), dtype=int)
            self.board = np.vstack([new_row, self.board])
        self.lines_cleared += len(full_rows)
        return len(full_rows)

    def render(self):
        '''Renders the current game board'''

        board_copy = self.board.copy()

        piece_height, piece_width = self.curr_piece.shape
        for i in range(piece_height):
            for j in range(piece_width):
                if self.curr_piece[i, j] == 1:
                    x = self.curr_pos[0] + j
                    y = self.curr_pos[1] + i
                    if 0 <= x < Tetris.BOARD_WIDTH and 0 <= y < Tetris.BOARD_HEIGHT:
                        board_copy[y, x] = 1  # Overlay the falling Tetromino

        # Convert board to cv2 image
        img = np.array([[Tetris.COLORS[cell] for cell in row]
                       for row in board_copy], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB').resize(
            (Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)

        img = np.array(img)
        # Add the score text to the image
        score_text = f"Score: {self.score}"
        cv2.putText(
            img,
            score_text,
            (10, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # Show the game using OpenCV
        cv2.imshow('Tetris', np.array(img))
        cv2.waitKey(1)

    # TESTING

    def set_curr(self, tetromino):
        '''Sets the current piece to tetromino. Used for debugging'''

        self.curr_piece = Tetris.TETROMINOS[tetromino]

    def set_board(self, nboard):
        '''Sets the current board to nboard. Used for debugging'''
        self.board = np.array(nboard)

    def get_score(self):
        return self.score

    # STATISTICS

    def calculate_holes(self):
        '''
        Given a board, calculate the number of holes that exist within the board.
        A "hole" is defined when there exists an empty pixel and there exists a placed pixel above it in the same column.
        '''

        holes = 0
        col_holes = [0 for _ in range(Tetris.BOARD_HEIGHT)]

        for x in range(Tetris.BOARD_HEIGHT):
            for y in range(Tetris.BOARD_WIDTH):
                if self.board[x, y] == 0:
                    if 1 in self.board[:x, y]:
                        # print(board[:y, x])
                        col_holes[x] += 1

        # print(col_holes)
        holes = sum(col_holes)
        return holes

    def calculate_aggregated_height(self):
        ''' Computes the aggregated height and returns it.'''
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

    def get_next_states(self):
        '''
        Finds every possible valid move that you can do with the current state of the board. Includes the current column and rotation
        '''

        states = {}

        # Itearate all 4 rotations including 0 rotation
        for rotation in range(4):

            rotated_piece = np.rot90(self.curr_piece, rotation)

            # Place piece in every column and calculate the board state

            for col in range(Tetris.BOARD_WIDTH):
                # Init temporary board as default before evaluating
                temp_board = self.board.copy()
                pos = [col, 0]

                # Drop piece until collision
                while not self.check_collision(rotated_piece, pos):
                    pos[1] += 1
                pos[1] -= 1  # Move back up after the collision

                # Check if final position is valid
                if not self.check_collision(rotated_piece, pos):
                    self.board = self.add_piece(rotated_piece, pos)

                    # Add to possible states
                    states[(col, rotation)] = self.get_board_properties()

        return states

    def get_board_properties(self):
        """
        Returns all statistics of curent board and properties
        """

        lines_cleared = self.lines_cleared
        aggregated_height = self.calculate_aggregated_height()
        total_holes = self.calculate_holes()
        bumpiness = self.calculate_bumpiness()

        return [lines_cleared, aggregated_height, total_holes, bumpiness]

    # DQN

    def step(self, action, render=True):
        x_pos, rotation = action
        # Check if shape is valid to place based on its width and if it may cross boundaries
        if x_pos < 0 or x_pos + self.curr_piece.shape[1] > Tetris.BOARD_WIDTH:
            return
        self.curr_pos = [x_pos, 0]
        self.rotate_piece(rotation)
        while not self.check_collision(self.curr_piece, self.curr_pos):
            self.curr_pos[1] += 1
        self.curr_pos[1] -= 1
        self.board = self.add_piece(self.curr_piece, self.curr_pos)
        self.update_score()
        if render:
            self.render()
        if not self.game_over:
            self.spawn_piece()
        else:
            self.score -= 2
        return self.score, self.game_over

    def calculate_bumpiness(self):
        '''
        Given a board, calculate the difference of heights between two adjacent columns.
        An undesirable board is one where there exists deep "wells"
        '''
        bumpiness = 0
        col_heights = [0 for _ in range(Tetris.BOARD_WIDTH)]

        for x in range(Tetris.BOARD_WIDTH):
            for y in range(Tetris.BOARD_HEIGHT):
                # Finds the nearest pixel in iterated column and find its height before breaking and iterating to next column.
                if self.board[y, x] == 1:
                    col_heights[x] = Tetris.BOARD_HEIGHT - y
                    break

        # print(col_heights)
        for idx in range(1, len(col_heights)):
            bumpiness += abs(col_heights[idx]-col_heights[idx-1])

        return bumpiness

# Testing


if __name__ == "__main__":
    game = Tetris()

    # board1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #           [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    #           [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    #           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #           [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    #           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    # game.set_board(board1)
    # print(game.board)
