import random
import copy
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def min_value(self, state, depth, alpha, beta):
        if abs(self.game_value(state)) == 1:
            return self.game_value(state)

        if depth == self.depth_limit:
            return self.heuristic_game_value(state)

        suc = self.succ(state, self.opp) # get all the possible successors
        for succ in suc:
            state_copy = copy.deepcopy(state)
            if self.get_drop_phase(state):
                state_copy[succ[0]][succ[1]] = self.opp
            else:
                state_copy[succ[0]][succ[1]] = ' '
                state_copy[succ[2]][succ[3]] = self.opp 
            
            beta = min(beta, self.max_value(state_copy, depth+1, alpha, beta))

        if beta <= alpha:
            return alpha

        return beta


    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.depth_limit = 2


    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        sc = float("-inf")
        move = []
        move2 = None
        old3 = None

        suc = self.succ(state, self.my_piece)
        for succ in suc:
            drop_phase = self.get_drop_phase(state)
            state_copy = copy.deepcopy(state)

            if drop_phase:
                state_copy[succ[0]][succ[1]] = self.my_piece
            else:
                state_copy[succ[0]][succ[1]] = ' '
                state_copy[succ[2]][succ[3]] = self.my_piece

            possible_score = self.max_value(state_copy, 0, float("-inf"), float("inf"))

            if possible_score > sc:
                sc = possible_score
                move2 = (succ[2], succ[3])
                if not drop_phase:
                    old3 = (succ[0], succ[1])

        if old3 is not None:
            move.append(move2)
            move.append(old3)
        else:
            move.append(move2)

        return move


    def get_adj_moves(self, state, row, col):
        dim = 5
        moves = []

        if row + 1 < dim and state[row+1][col] == ' ': 
            moves.append((row+1, col))
        if row - 1 >= 0 and state[row-1][col] == ' ':
            moves.append((row-1, col))
        if col + 1 < dim and state[row][col+1] == ' ': 
            moves.append((row, col+1))
        if col - 1 >= 0 and state[row][col-1] == ' ': 
            moves.append((row, col-1))

        if row - 1 >= 0 and col - 1 >= 0 and state[row-1][col-1] == ' ': 
            moves.append((row-1, col-1))
        if row - 1 >= 0 and col + 1 < dim and state[row-1][col+1] == ' ': 
            moves.append((row-1, col+1))
        if row + 1 < dim and col - 1 >= 0 and state[row+1][col-1] == ' ': 
            moves.append((row+1, col-1))
        if row + 1 < dim and col + 1 < dim and state[row+1][col+1] == ' ': 
            moves.append((row+1, col+1))

        return moves


    def get_drop_phase(self, state):
        count = 0
        for r in range(5):
            for c in range(5):
                if state[r][c] != ' ':
                    count += 1

        return count < 8 


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        self.place_piece(move, self.opp)


    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece


    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def max_value(self, state, depth, alpha, beta):
        if abs(self.game_value(state)) == 1:
            return self.game_value(state)

        if depth == self.depth_limit:
            return self.heuristic_game_value(state)

        suc = self.succ(state, self.my_piece) # get all the possible successors
        for succ in suc:
            state_copy = copy.deepcopy(state)
            if self.get_drop_phase(state):
                state_copy[succ[0]][succ[1]] = self.my_piece # place piece
            else:
                state_copy[succ[0]][succ[1]] = ' '
                state_copy[succ[2]][succ[3]] = self.my_piece # move exisiting piece
            
            alpha = max(alpha, self.min_value(state_copy, depth+1, alpha, beta)) # update the alpha

        if alpha >= beta:
            return beta

        return alpha


    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        for col in range(2):
            for i in range(0,2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col+1] == state[i+2][col+2] == state[i+3][col+3]:
                    return 1 if state[i][col] == self.my_piece else -1
        
        for col in range(2):
            for i in range(3,5):
                if state[i][col] != ' ' and state[i][col] == state[i-1][col+1] == state[i-2][col+2] == state[i-3][col+3]:
                    return 1 if state[i][col] == self.my_piece else -1
        
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row][col+1] == state[row+1][col+1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # no winner yet


    def heuristic_game_value(self, state):
        weight = [
            [0.05, 0.1, 0.05, 0.1, 0.05],
            [ 0.1, 0.2,  0.2, 0.2, 0.1],
            [0.05, 0.2,  0.3, 0.2, 0.05],
            [ 0.1, 0.2,  0.2, 0.2, 0.1],
            [0.05, 0.1, 0.05, 0.1, 0.05]
        ]
        val = self.game_value(state)
        
        if val != 0: 
            return val
        
        player_score = 0
        opp_score = 0
        for row in range(5):
            for col in range(5):
                if state[row][col] == self.my_piece:
                    player_score += weight[row][col]
                elif state[row][col] == self.opp:
                    opp_score += weight[row][col]

        return player_score - opp_score

    def succ(self, state, teamcolor):
        suc = []
        drop_phase = self.get_drop_phase(state)
        
        if drop_phase:
           # any empty cell is a possible location for a piece which is a successor
           for r in range(5):
               for c in range(5):
                   if state[r][c] == ' ':
                       suc.append([r, c, r, c])
        else:
            # look for the cells with pieces and check the adjacent moves
            for r in range(5):
                for c in range(5):
                    if state[r][c] == teamcolor:
                        moves = self.get_adj_moves(state, r, c)
                        for move in moves:
                            suc.append([r, c, move[0], move[1]])

        return suc



############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
