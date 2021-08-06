import numpy as np
import sys

from controller import TQController, QLController


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    # graph = np.zeros((12,12))
    graph = np.array([
        [0] * 12,
        [0] + [1] * 2 + [0] + [1] *7 + [0],
        [0] + [1] * 2 + [0] + [1] + [0] + [1]*2 + [0] * 2 + [1]+ [0],
        [0] * 2 + [1] + [0] + [1] + [0]*4 + [1] * 2 + [0],
        [0] + [1] *5 + [0] + [1]*4 + [0],
        [0] + [1] *5 + [0] + [1]*4 + [0],
        [0] * 5 + [1] + [0] + [1] + [0] + [1] *2 + [0],
        [0] + [1] *5 + [0] + [1] + [0] + [1]*2 + [0],
        [0] + [1] + [0]* 5 + [1] + [0] + [1]*2 + [0],
        [0] + [1] * 4 + [0] + [1]*2 + [0] + [1]*2 + [0],
        [0] + [1] * 2 + [0] + [1]*4 + [0] + [1]*2 + [0],
        [0] * 12
    ])
    start = [1,1]
    goal = [10, 10]
    
    # num_episode = 10000
    params = {
        "num_episodes": 50,
        "alpha": 0.05,
        "epsilon": 0.05,
        "gamma": 0.97
    }
    # cntr = TQController(graph, start, goal, **params)
    cntr = QLController(graph, start, goal, **params)

    cntr.play()
    
    