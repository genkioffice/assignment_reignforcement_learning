from environment import TQEnvironment
import numpy as np
class BasicEvaluator:
    '''
        The evaluator class saves the value table V and it controls how to update the V.
        The V is updated from an agent class.
    '''
    def __init__(self):
        # self.V = np.zeros(1)
        pass

class TQEvaluator(BasicEvaluator):
    def __init__(self, size, alpha, gamma):
        super().__init__()
        self.V = np.zeros((size[1],size[0]))
        self.alpha = alpha
        self.gamma = gamma
    
    def tq(self, reward, px,py,cx,cy):
        '''
            TQ updates.
            parameters:
                px: int.
                    Previous position of x.
                py: int.
                    Previous position of y.
                cx: int.
                    Current position of x.
                cy: int.
                    Current position of y.
            
            returns: Null.
        '''
        self.V[py][px] += (-self.alpha) * self.V[py][px] + self.alpha * (reward + self.gamma * self.V[cy][cx])

    
    def get_value(self, cx, cy):
        return self.V[cy][cx]

    def save_pef(self, val):
        self.performance = val

    def get_pef(self):
        return self.performance

class QLEvaluator(TQEvaluator):
    def __init__(self, size, alpha, gamma):
        super().__init__(size, alpha, gamma)
        self.V = np.zeros((size[1],size[0]))
        self.rx = [1,-1,0,0]
        self.ry = [0,0,1,-1]


    def tq(self, reward, px, py, cx, cy, env:TQEnvironment):
        # V[cy][cx] is given and the next action is calculated
        nx_max_q = self.calu_nx_q(cx, cy, env)
        # print(reward + self.gamma * nx_max_q)
        self.V[py][px] += -self.alpha * self.V[py][px] + self.alpha * (reward + self.gamma * nx_max_q)

    def calu_nx_q(self, cx, cy, env:TQEnvironment):
        nx_qs = []
        for i in np.arange(0,4):
            x = cx + self.rx[i]
            y = cy + self.ry[i]
            # check feasibility
            if env.is_invalid(x, y):
                nx_qs.append(-float('inf'))
            else:
                nx_qs.append(self.V[y][x])
        q = np.max(nx_qs)
        return q
