
import numpy as np
from environment import TQEnvironment
from evaluator import TQEvaluator

class BasicPolicy:
    '''
        The Policy determines how the next action generate.
    '''
    def __init__(self):
        self.actions = []

    def next_action(self):
        pass

class EGreedyPolicy(BasicPolicy):
    def __init__(self, epsilon, gamma) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.rx = [1,-1,0,0]
        self.ry = [0,0,1,-1]


    def next_action(self, px, py, env:TQEnvironment, evaluator:TQEvaluator):
        '''
            This function reuturns next cordinates by following epsilon-greedy algorithm.
        '''
        e = int(np.random.binomial(1, self.epsilon, 1))
        # explore
        if e == 1:
            idx = int(np.random.randint(0,4,1))
            x = px + self.rx[idx]
            y = py + self.ry[idx]
            return x, y
        # utilize
        else:
            pos = []
            for i in np.arange(0,4):
                x = px + self.rx[i]
                y = py + self.ry[i]
                # check feasibility
                if env.is_invalid(x, y):
                    pos.append(-float('inf'))
                else:
                    pos.append(self.gamma * evaluator.get_value(x, y) + env.fast_reward(x, y))
            pos = [i for i, v in enumerate(pos) if v >= np.max(pos)]
            # randomize argmaxes to avoid be trapped
            np.random.shuffle(pos)
            idx = pos[0]
            x = px + self.rx[idx]
            y = py + self.ry[idx]
            # print(pos)
            # print(y, x)
            return x, y