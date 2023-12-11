import heapq


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader.

    INPUT:
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    man_dist_sum = 0
    for s in range(9):
        if from_state[s] == 0:
            continue
        man_dist_sum += abs(s // 3 - (from_state[s] - 1) // 3) + abs(
            s % 3 - (from_state[s] - 1) % 3
        )
    return man_dist_sum


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT:
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle.
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    successor_states = []
    pos_of_empty_tile = []

    for x in range(len(state)):
        if state[x] == 0:
            pos_of_empty_tile.append(x)

    for x in pos_of_empty_tile:
        following = state.copy()

        if x == 0:
            if following[1] != 0:
                following[0], following[1] = following[1], following[0]
                successor_states.append(following)
                following = state.copy()

            if following[3] != 0:
                following[0], following[3] = following[3], following[0]
                successor_states.append(following)

        elif x == 1:
            if following[0] != 0:
                following[0], following[1] = following[1], following[0]
                successor_states.append(following)
                following = state.copy()

            if following[2] != 0:
                following[2], following[1] = following[1], following[2]
                successor_states.append(following)
                following = state.copy()

            if following[4] != 0:
                following[4], following[1] = following[1], following[4]
                successor_states.append(following)

        elif x == 2:
            if following[1] != 0:
                following[2], following[1] = following[1], following[2]
                successor_states.append(following)
                following = state.copy()

            if following[5] != 0:
                following[2], following[5] = following[5], following[2]
                successor_states.append(following)

        elif x == 3:
            if following[0] != 0:
                following[0], following[3] = following[3], following[0]
                successor_states.append(following)
                following = state.copy()

            if following[6] != 0:
                following[6], following[3] = following[3], following[6]
                successor_states.append(following)
                following = state.copy()

            if following[4] != 0:
                following[4], following[3] = following[3], following[4]
                successor_states.append(following)

        if x == 4:
            if following[1] != 0:
                following[1], following[4] = following[4], following[1]
                successor_states.append(following)
                following = state.copy()

            if following[3] != 0:
                following[3], following[4] = following[4], following[3]
                successor_states.append(following)
                following = state.copy()

            if following[5] != 0:
                following[5], following[4] = following[4], following[5]
                successor_states.append(following)
                following = state.copy()

            if following[7] != 0:
                following[7], following[4] = following[4], following[7]
                successor_states.append(following)

        elif x == 5:
            if following[2] != 0:
                following[2], following[5] = following[5], following[2]
                successor_states.append(following)
                following = state.copy()

            if following[8] != 0:
                following[8], following[5] = following[5], following[8]
                successor_states.append(following)
                following = state.copy()

            if following[4] != 0:
                following[4], following[5] = following[5], following[4]
                successor_states.append(following)

        elif x == 6:
            if following[3] != 0:
                following[6], following[3] = following[3], following[6]
                successor_states.append(following)
                following = state.copy()

            if following[7] != 0:
                following[6], following[7] = following[7], following[6]
                successor_states.append(following)

        elif x == 7:
            if following[8] != 0:
                following[8], following[7] = following[7], following[8]
                successor_states.append(following)
                following = state.copy()

            if following[6] != 0:
                following[6], following[7] = following[7], following[6]
                successor_states.append(following)
                following = state.copy()

            if following[4] != 0:
                following[4], following[7] = following[7], following[4]
                successor_states.append(following)

        elif x == 8:
            if following[5] != 0:
                following[8], following[5] = following[5], following[8]
                successor_states.append(following)
                following = state.copy()

            if following[7] != 0:
                following[8], following[7] = following[7], following[8]
                successor_states.append(following)

    return sorted(successor_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT:
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    openq = []
    closed_set = []
    closed = []
    parents = []
    path = []
    g = 0
    heapq.heappush(
        openq,
        (
            get_manhattan_distance(state) + g,
            state,
            (g, get_manhattan_distance(state), -1),
        ),
    )
    max_length = 1
    max_length = 0
    while openq != []:
        max_length += 1
        parent = heapq.heappop(openq)
        parents.append(parent)
        closed.append(parent[1])
        heapq.heappush(closed_set, parent)
        if parent[1] == goal_state:
            path.append(parent)
            while parent[2][2] != -1:
                parent = parents[parent[2][2]]
                path.append(parent)
            reverse = path[::-1]
            m = 0
            for x in reverse:
                print(x[1], "h={}".format(x[2][1]), "moves:", m)
                m += 1
            break
        successor_states = get_succ(parent[1])
        g = parent[2][0] + 1
        for ss in successor_states:
            if ss in closed:
                continue
            else:
                h = get_manhattan_distance(ss)
                to_add = (h + g, ss, (g, h, parents.index(parent)))
                heapq.heappush(openq, to_add)
        max_length = max(max_length, len(openq))
    print("Max queue length: {}".format(max_length))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions.
    Note that this part will not be graded.
    """
    # solve([3, 4, 6, 0, 0, 1, 7, 2, 5])  # 46
    # solve([6, 0, 0, 3, 5, 1, 7, 2, 4])  # 672
    # solve([0, 4, 7, 1, 3, 0, 6, 2, 5])  # 445
