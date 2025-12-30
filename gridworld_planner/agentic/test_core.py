"""
Basic tests for Gridworld core components.
"""
import pytest
from agentic.env.gridworld import Gridworld
from agentic.env.generators import generate_obstacles
from agentic.search.astar import astar_search
from agentic.search.bfs import bfs_search
from agentic.search.mcts import mcts_search
from agentic.search.heuristics import manhattan

def test_gridworld_neighbors():
    # Test that the Gridworld neighbor function returns correct neighbors
    env = Gridworld(5, 5, [], (0, 0), (4, 4))
    assert set(env.neighbors((0, 0))) == {(1, 0), (0, 1)}
    assert set(env.neighbors((2, 2))) == {(1, 2), (3, 2), (2, 1), (2, 3)}

def test_astar_and_bfs_find_path():
    # Test that A* and BFS both find a valid path of the same length
    env = Gridworld(5, 5, [], (0, 0), (4, 4))
    path_a, *_ = astar_search(env, manhattan)
    path_b, *_ = bfs_search(env)
    assert path_a is not None and path_b is not None
    assert path_a[0] == (0, 0) and path_a[-1] == (4, 4)
    assert path_b[0] == (0, 0) and path_b[-1] == (4, 4)
    assert len(path_a) == len(path_b)

def test_mcts_finds_path():
    # Test that MCTS finds a valid path from start to goal
    env = Gridworld(5, 5, [], (0, 0), (4, 4))
    path, *_ = mcts_search(env, max_iterations=5000, rollout_depth=20, timeout_ms=2000)
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (4, 4)

def test_obstacle_generation():
    # Test that obstacles are generated away from start and goal, and count is reasonable
    obs = generate_obstacles(5, 5, 0.2, 42, (0, 0), (4, 4))
    assert all(o != (0, 0) and o != (4, 4) for o in obs)
    assert len(obs) <= 25 * 0.2 + 1
