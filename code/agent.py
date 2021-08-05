import numpy as np
from evaluator import TQEvaluator
from environment import TQEnvironment

class BasicAgent:
    '''
        This class plays as an agent. The main purpose of the agent is to update V in Evaluator.
        This class is comprised of a policy which determines how the next action generate, a evaluator
        which has the V and controls how to update it.
    '''
    def __init__(self, policy) -> None:
        self.policy = policy
        # self.V = np.zeros(1)
        pass
    
    def update(self):
        pass

class TQAgent(BasicAgent):
    def __init__(self, policy) -> None:
        super().__init__(policy)
        self.performance = 0 # the number of iteration

    def update(self, px, py, evaluator:TQEvaluator, env:TQEnvironment):
        '''
            This method takes an environment object to decide the generated action is feasible, and check 
            it satisfies the stop condition.
        '''
        self.performance += 1
        x,y = self.policy.next_action(px, py, env, evaluator)
        while env.is_invalid(x,y):
            x, y = self.policy.next_action(px, py, env, evaluator)
        reward = env.fast_reward(x,y)
        evaluator.tq(reward, px, py, x, y)
        if env.is_goal(x,y):
            return self.performance
        return self.update(x, y, evaluator, env)