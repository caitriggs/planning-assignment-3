import numpy as np
import math
import random
from game import BoardState, GameSimulator

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()
        actions = self.sim.generate_valid_actions(p)
        
        # Debugging output to check if valid actions exist
        if not actions:
            print(f"DEBUG: No valid actions available for player {p} in state {state}")
        else:
            print(f"DEBUG: Available actions for player {p}: {actions}")
        
        return actions
    
    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        print(f"execute state: {state}")
        print(f"execute action: {action}")
        next_state = tuple((s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2)
        print(f"Next state generated: {next_state}")

    def is_goal(self, state):
        """
        Checks if the current state is a goal state (i.e., either player has won).
        White wins if their ball is in row 7 (positions 49-55),
        Black wins if their ball is in row 0 (positions 0-6).
        """
        board, player_idx = state
        white_ball = board[5]  # White ball position
        black_ball = board[11]  # Black ball position

        # White wins if the ball is in row 7 (positions 49-55)
        if 49 <= white_ball <= 55:
            return True
        # Black wins if the ball is in row 0 (positions 0-6)
        if 0 <= black_ball <= 6:
            return True
        # If neither has won, return False
        return False
    
    # def is_goal(self, state):
    #     """
    #     Checks if the state is a goal state in the set of goal states
    #     """
    #     return state in self.goal_state_set


class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """
        self.search_alg_fnc = self.monte_carlo_tree_search

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ##
    ## NOTE: Here is an example of the format:
    ##       [(s1, a1),(s2, a2), (s3, a3), ..., (sN, aN)] where
    ##          sN is an element of self.goal_state_set
    ##          aN is None
    ##          All sK for K=1...N are in the form (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
    ##              effectively encoded_state is the result of tuple(BoardState.state)
    ##          All aK for K=1...N are in the form (int, int)
    ##
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 

    def monte_carlo_tree_search(self, state_tup, *args):
        """
        Monte Carlo Tree Search algorithm for decision making.
        Parameters:
            - state_tup: (encoded_state, player_idx)
        Returns:
            - The best action determined by the MCTS algorithm.
        """
        root = MonteCarloTreeSearchNode(state=state_tup)

        # Set simulation limit
        simulation_limit = 10  # Number of iterations of MCTS

        print(f"state_tup: {state_tup}")
        for _ in range(simulation_limit):
        
            # 1. Selection: Select the most promising node using UCB
            node = root
            while node.is_fully_expanded():
                print(f"Step in sim {_} and is_fully_expanded")
                node = node.best_child()

            # 2. Expansion: Expand node (add children if it's not terminal)
            print(f"is_goal: {self.is_goal(node.state)}")
            print(f"is_termination_state: {self.sim.game_state.is_termination_state()}")
            print(f"node.state: {node.state}")
            if not self.is_goal(node.state) and not self.sim.game_state.is_termination_state():
                node.expand(self)  # Make sure expansion happens here

            # 3. Simulation: Run a random simulation from the expanded node
            result = node.simulate(self)

            # 4. Backpropagation: Propagate the result back up the tree
            node.backpropagate(result)

        # After simulations, choose the best action from the root node
        if root.children:
            best_child = root.best_child(exploration_weight=0)  # Best child based on win ratio
            return best_child.action, None
        else:
            raise ValueError("No valid moves to choose from. Root node has no children.")

    def heuristic(self, state):
        """
        Heuristic evaluation: prioritize states where the ball is closer to the winning row.
        """
        board, player_idx = state
        white_ball = board[5]  # White ball position
        black_ball = board[11]  # Black ball position

        # Evaluate based on the distance of the ball to the goal row.
        if player_idx == 0:  # White's turn
            return 7 - (white_ball // 7)  # Closer to row 7 is better for white
        else:  # Black's turn
            return black_ball // 7  # Closer to row 0 is better for black


class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.action = action

    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            raise ValueError("No children to choose from. Node has not been expanded.")
        
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        print(f"choices_weights: {choices_weights}")
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, problem):
        actions = problem.get_actions(self.state)
        if not actions:
            print("No valid actions found during expansion.")
        else:
            print(f"Found {len(actions)} valid actions: {actions}")

        for action in actions:
            next_state = problem.execute(self.state, action)
            if next_state:
                print(f"Action {action} led to next state: {next_state}")
                child_node = MonteCarloTreeSearchNode(state=next_state, parent=self, action=action)
                self.children.append(child_node)
                print(f"Child node added for action {action}. Total children: {len(self.children)}")
            else:
                print(f"Action {action} did not generate a valid next state.")

    def simulate(self, problem):
        state = self.state
        current_player = state[1]
        while not problem.is_goal(state) and not problem.sim.game_state.is_termination_state():
            possible_actions = problem.get_actions(state)
            if not possible_actions:
                break
            action = random.choice(possible_actions)  # Randomly select action
            state = problem.execute(state, action)
            current_player = (current_player + 1) % 2  # Switch player
        print(f"problem.heuristic(state): {problem.heuristic(state)}")
        return problem.heuristic(state)

    def backpropagate(self, result):
        self.visits += 1
        if result > 0:  # Assuming result > 0 means a win for maximizing player
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)


if __name__ == '__main__':
    from game import BoardState, GameSimulator, PlayerAlgorithmA, PlayerAlgorithmB, AdversarialSearchPlayer

    def test_adversarial_search(p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        b1 = BoardState()
        b1.state = np.array(encoded_state_tuple)
        b1.decode_state = b1.make_state()
        players = [p1_class(GameStateProblem(b1, b1, 0), 0), p2_class(GameStateProblem(b1, b1, 0), 1)]
        sim = GameSimulator(players)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        assert winner == exp_winner and status == exp_stat

    test_adversarial_search(PlayerAlgorithmA, PlayerAlgorithmB, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues")

    # tests = [
    #         (PlayerAlgorithmA, PlayerAlgorithmB, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    #         (AdversarialSearchPlayer, PlayerAlgorithmB, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"), 
    #         (AdversarialSearchPlayer, PlayerAlgorithmB, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"), 
    #         (PlayerAlgorithmB, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22,  9, 20, 34, 39, 55, 55), "BLACK", "No issues"), 
    #         (PlayerAlgorithmB, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 11, 20, 34, 39, 55, 55), "BLACK", "No issues"), 
    #         (AdversarialSearchPlayer, PlayerAlgorithmB, (44, 37, 46, 34, 40, 34,  1,  2, 52,  4,  5, 52), "WHITE", "No issues"), 
    #         (AdversarialSearchPlayer, PlayerAlgorithmB, (44, 37, 46, 28, 40, 28,  1,  2, 52,  4,  5, 52), "WHITE", "No issues") 
    #     ]
    
    # for test in tests:
    #     test_adversarial_search(*test)
