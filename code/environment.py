class BasicEnvironment:
    '''
        The Environment class check whether the generate action is valid.
        Plus, it exhibits the reward given an action.
    '''
    def __init__(self) -> None:
        pass
    
    def fast_reward(self) -> float:
        pass


class TQEnvironment(BasicEnvironment):
    def __init__(self, graph, goal, start=(0,0)) -> None:
        '''
            parameters:
                goal: tuple.
                    The goal of the exploration. Ex) (8,9)
                
        '''
        super().__init__()
        self.graph = graph
        self.start = start
        self.goal = goal
    
    def is_invalid(self, x, y):
        if (self.graph.shape[0] <= y) | (self.graph.shape[1] <= x) | (x < 0) | (y < 0):
            return True
        if (self.graph[y][x] == 0):
            return True
        return False

    def is_goal(self, x, y):
        if (x == self.goal[1]) & (y == self.goal[0]):
            return True
        else:
            return False
    
    def fast_reward(self, x, y):
        if self.is_goal(x, y):
            return 1
        else:
            return 0