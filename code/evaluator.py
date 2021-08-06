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