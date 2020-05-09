import numpy as np, math, random, time
from itertools import permutations 

move_map = {
    0: ' ',
    1: 'X',
    -1: 'O'
}

class board:

    def __init__(self):
        self.state = np.zeros((6,7)).astype('int')
        self.available_columns = np.arange(7).astype('int')

    #method to display the board in a human readable format
    def show_board(self):
        line = '---------------'
        print(line)
        for i in range(len(self.state)):
            output = '|'
            for j in range(len(self.state[i])):
                output += move_map[self.state[i][j]] + '|'
            print(output)
            print(line)

    #check if the game is won
    def check_win(self):
        update = np.arange(4).astype('int')
        constant = np.zeros(4).astype('int')
        changes = [[update, constant],[constant, update],[update, update],[update, -update]]
        for row in range(6):
            for col in range(7):
                for change in changes:
                    rows = change[0] + row
                    cols = change[1] + col
                    if max(rows) > 5: continue
                    if max(cols) > 6: continue
                    if min(rows) < 0: continue
                    if min(cols) < 0: continue
                    indices = list(zip(rows, cols))
                    #print(indices)
                    vals = [self.state[indices[i]] for i in range(4)]
                    val = sum(vals)
                    if val == 4:
                        return 1
                    if val == -4:
                        return -1

        return 0

    #check for a win or tie
    def game_over(self):
        if len(self.available_columns) == 0:
            return True
        outcome = self.check_win()
        if outcome == 0:
            return False
        else:
            return True

    def do_move(self, column, player):
        if column not in self.available_columns:
            return -1
        for row in range(5,-1,-1):
            if self.state[row][column] == 0:
                self.state[row][column] = player
                break
        if row == 0:
            self.available_columns = np.delete(self.available_columns, np.argwhere(self.available_columns == column))

    def copy(self):
        copy = board()
        copy.state = self.state.copy()
        copy.available_columns = self.available_columns.copy()
        return copy

    def flip(self):
        copy = board()
        copy.state = np.flip(self.state.copy(), axis=1)
        columns = []
        for col in range(7):
            if copy.state[0,col] == 0:
                columns.append(col)
        copy.available_columns = np.array(columns)
        return copy

    def switch(self):
        copy = board()
        copy.state = self.state.copy() * -1
        copy.available_columns = self.available_columns.copy()
        return copy

    def generate_channels(self):
        stream = np.zeros((6,7,3))
        for i in range(3):
            stream[:,:,i] = self.state == (i-1)