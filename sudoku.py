import numpy as np
import collections
import itertools
import pandas as pd

# Part 1:
def string_puzzle_to_arr(puzzle):
    return np.array([list(line.strip()) for line in puzzle.split('\n') if line.strip()], dtype=np.int)

class Board:
    def __init__(self, board):
        # If board is a string, convert it to a 9x9 NumPy array
        if isinstance(board, str):
            # Remove any newline characters
            board = board.replace('\n', '')
            # Convert the cleaned string into a 9x9 array of integers
            board = [list(map(int, board[i:i + 9])) for i in range(0, len(board), 9)]
            self.arr = np.array(board)
        # If board is already a NumPy array
        elif isinstance(board, np.ndarray):
            self.arr = board

    def get_row(self, row_idx):
        return self.arr[row_idx]

    def get_column(self, col_idx):
        return self.arr[:, col_idx]

    def get_block(self, pos_1, pos_2):
        start_row = pos_1 * 3
        start_col = pos_2 * 3
        return self.arr[start_row:start_row + 3, start_col:start_col + 3]
    
    def iter_rows(self):
        return [self.arr[i, :] for i in range(9)]

    def iter_columns(self):
        return [self.arr[:, i] for i in range(9)]

    def iter_blocks(self):
        blocks = []
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                blocks.append(self.arr[row:row + 3, col:col + 3])
        return blocks


# Part 2:
def is_subset_valid(arr):
    check = []
    for elem in arr.flatten():  # Flatten the array to handle both rows, columns, and blocks
        if elem != 0:  # Ignore zeros as they represent empty cells
            if elem in check:
                return False
            check.append(elem)
    return True 

def is_valid(board):
    # Check all rows
    for i in range(9):
        if not is_subset_valid(board.get_row(i)):
            return False
    
    # Check all columns
    for i in range(9):
        if not is_subset_valid(board.get_column(i)):
            return False
    
    # Check all 3x3 blocks
    for i in range(3):
        for j in range(3):
            if not is_subset_valid(board.get_block(i, j)):
                return False
    
    return True


# Part 3:
def find_empty(board):
    empty_positions = []

    # Iterate through each cell in the 9x9 board
    for i in range(9):
        for j in range(9):
            if board.arr[i, j] == 0:  # Check if the cell is empty
                empty_positions.append((i, j))  # Add the position as a tuple

    # Convert the list of tuples to a NumPy array
    return np.array(empty_positions)



def is_full(board):
    empty_positions = []
    # Iterate through each cell in the 9x9 board
    for i in range(9):
        for j in range(9):
            if board.arr[i, j] == 0:  # Check if the cell is empty
                empty_positions.append((i, j))  # Add the position as a tuple
    if len(empty_positions) > 0:
        return False
    else:
        return True 


def find_possibilities(board, x, y):
    # Access the underlying NumPy array
    arr = board.arr
    
    if arr[x, y] != 0:
        return []  # If the cell is already filled, there are no possibilities.

    possibilities = set(range(1, 10))  # Possible values are 1 to 9.

    # Remove values present in the same row.
    possibilities -= set(arr[x, :])

    # Remove values present in the same column.
    possibilities -= set(arr[:, y])

    # Determine the 3x3 block.
    start_row, start_col = 3 * (x // 3), 3 * (y // 3)
    block = arr[start_row:start_row + 3, start_col:start_col + 3]

    # Remove values present in the 3x3 block.
    possibilities -= set(block.flatten())

    return list(possibilities)


# Part 4:
def adapt_long_sudoku_line_to_array(line):
    grid = [int(num) for num in line]
    # Reshape the list into a 9x9 NumPy array
    return np.array(grid).reshape(9, 9)


def read_sudokus_from_csv(file_name, read_solutions=False):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)
    
    # If read_solutions is True, read the second column (solutions)
    if read_solutions:
        column_index = 1
    else:
        column_index = 0  # Read the first column (empty puzzles)
    
    sudokus = df.iloc[:, column_index]
    
    sudoku_grids = [np.array([int(num) for num in sudoku]).reshape(9, 9) for sudoku in sudokus]
    
    return sudoku_grids


def is_valid_sudoku(grid):
    # Check if a block (row, column, or subgrid) is valid
    def is_valid_block(block):
        block = [num for num in block if num != 0]  # Ignore zeros
        return len(block) == len(set(block))  # Check for duplicates
    
    # Check rows
    for row in grid:
        if not is_valid_block(row):
            return False
    
    # Check columns (by transposing the grid)
    for col in grid.T:
        if not is_valid_block(col):
            return False
    
    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = grid[i:i+3, j:j+3].flatten()
            if not is_valid_block(subgrid):
                return False
    
    return True

def detect_invalid_solutions(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    
    # Extract the solutions from the second column (index 1)
    solutions = df.iloc[:, 1]
    
    invalid_sudokus = []
    
    # Convert each solution string into a 9x9 NumPy array and validate it
    for solution in solutions:
        grid = np.array([int(num) for num in solution]).reshape(9, 9)
        
        # Check if the Sudoku solution is invalid
        if not is_valid_sudoku(grid):
            invalid_sudokus.append(grid)
    
    # Return as a single NumPy array of invalid Sudoku grids
    if invalid_sudokus:
        return np.array(invalid_sudokus)
    else:
        return None  # Or you can return np.array([]) if you want to return an empty array
