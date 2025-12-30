"""
Monte Carlo Tree Search (MCTS) for Gridworld (unit-cost, 4-neighbor).
 is a simple, fixed-policy MCTS for demonstration and ablation.
"""
import random
import time
from agentic.env.gridworld import Gridworld


# Node for MCTS tree
class MCTSNode:
    def __init__(self, state, parent=None):
        # Initialize a node in the MCTS tree
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    # Check if all children expanded
    def is_fully_expanded(self, env):
        # Returns True if all possible children have been expanded
        return len(self.children) == len(env.neighbors(self.state))

    # Pick best child using UCB formula - upper confidence bound
    # balance exploitation with exploration when choosing which child node to visit next
    def best_child(self, c_param=1.4):
        # Select the child with the highest UCB value
        choices = [child for child in self.children]
        if not choices:
            return None
        return max(choices, key=lambda n: n.value / (n.visits + 1e-8) + c_param * ( ( (self.visits + 1e-8) ** 0.5 ) / (n.visits + 1e-8) ))


# Main MCTS loop for planning
def mcts_search(env: Gridworld, max_iterations=1000, rollout_depth=40, timeout_ms=2000):
    # Run MCTS for a fixed number of iterations or until timeout
    start_time = time.time()
    root = MCTSNode(env.start)
    nodes_expanded = 0
    max_frontier_size = 1
    best_path = None
    best_cost = float('inf')
    for _ in range(max_iterations):
        node = root
        state = node.state
        # Selection: descend tree by best child
        while node.children and node.is_fully_expanded(env):
            node = node.best_child()
            state = node.state
        # Expansion: add a new child if possible
        untried = [s for s in env.neighbors(state) if all(child.state != s for child in node.children)]
        if untried:
            next_state = random.choice(untried)
            child = MCTSNode(next_state, parent=node)
            node.children.append(child)
            node = child
            state = next_state
        # Simulation: random rollout from new node
        sim_state = state
        path = [sim_state]
        for _ in range(rollout_depth):
            if env.is_goal(sim_state):
                break
            neighbors = env.neighbors(sim_state)
            if not neighbors:
                break
            sim_state = random.choice(neighbors)
            path.append(sim_state)
        # Backpropagation: update stats up the tree
        reward = -len(path) if env.is_goal(sim_state) else -1000
        temp = node
        while temp is not None:
            temp.visits += 1
            temp.value += reward
            temp = temp.parent
        nodes_expanded += 1
        max_frontier_size = max(max_frontier_size, len(node.children))
        if env.is_goal(sim_state) and len(path) < best_cost:
            # Reconstruct full path from root
            full_path = []
            n = node
            while n is not None:
                full_path.append(n.state)
                n = n.parent
            full_path.reverse()
            best_path = full_path + path[1:]
            best_cost = len(best_path) - 1
        if (time.time() - start_time) * 1000 > timeout_ms:
            break
    if best_path:
        return best_path, nodes_expanded, max_frontier_size, 'goal_reached'
    return None, nodes_expanded, max_frontier_size, 'timeout'
