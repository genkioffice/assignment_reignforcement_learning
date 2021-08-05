import numpy as np
import time

from policy import EGreedyPolicy
from evaluator import TQEvaluator
from agent import TQAgent
from environment import TQEnvironment

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

class Controller:
    '''
        The controller class controls initialization.
        The value table is initalized at once and by pouring agent,
        the table is updated. You can also tweak the number of agents.
    '''
    def __init__(self, graph, start, goal, **kwargs) -> None:
        self.graph = graph
        self.start = start
        self.goal = goal
        self.num_episodes = kwargs.get('num_episodes', 1000)
        self.alpha = kwargs.get('alpha', 0.6)
        self.epsilon = kwargs.get('epsilon', 0.3)
        self.gamma = kwargs.get('gamma', 0.5)
        
    def play(self):
        policy = EGreedyPolicy(self.epsilon, self.gamma)
        evaluator = TQEvaluator(self.graph.shape, self.alpha, self.gamma)
        env = TQEnvironment(self.graph, self.goal, self.start)
        performances = []
        for i in np.arange(self.num_episodes):
            agent = TQAgent(policy)
            tmp = agent.update(self.start[1], self.start[0], evaluator, env)
            performances.append(tmp)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        plt.figure(figsize=(6,4))
        plt.plot(performances)
        plt.title(f"salsa iterations(n = {self.num_episodes})")
        plt.savefig(f"../plot/salsa_iterations_{timestamp}.png")
        
        plt.figure(figsize=(8,8))
        sns.heatmap(evaluator.V)
        plt.title(f"salsa heatmap(n = {self.num_episodes})")
        plt.savefig(f"../plot/salsa_heat_{timestamp}.png")
        