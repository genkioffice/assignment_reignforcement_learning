import numpy as np
import sys

from evaluator import TQEvaluator
from environment import TQEnvironment


sys.setrecursionlimit(100000)

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
            evaluator.save_pef(self.performance)
        return x, y
        # if env.is_goal(x,y):
        #     return self.performance
        # return self.update(x, y, evaluator, env)

    def wrapper(self, px, py, evaluator, env):
        # i = 0
        while (~(env.is_goal(px, py))):
            # i+=1
            px, py = self.update(px, py, evaluator, env)
            # print(i)
            # print(px, py)
            if env.is_goal(px,py):
                # print ("goal")
                return 

class QLAgent(BasicAgent):
    def __init__(self, policy) -> None:
        super().__init__(policy)
        self.performance = 0

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
        evaluator.tq(reward, px, py, x, y, env)
        if env.is_goal(x,y):
            evaluator.save_pef(self.performance)
        return x, y
        # if env.is_goal(x,y):
        #     return self.performance
        # return self.update(x, y, evaluator, env)

    def wrapper(self, px, py, evaluator, env:TQEnvironment):
        # i = 0
        while (~(env.is_goal(px, py))):
            # i+=1
            px, py = self.update(px, py, evaluator, env)
            # print(i)
            # print(px, py)
            if env.is_goal(px,py):
                # print ("goal")
                return 


