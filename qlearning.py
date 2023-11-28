import numpy as np
import random

import beetle_simulator as simulator

class QLearning:
    def __init__(self,n,empty_sqrs,beetle_source):
        self.sim_train = simulator.Simulator(n,empty_sqrs,beetle_source)
        self.sim_test = simulator.Simulator(n,empty_sqrs,beetle_source)
        self.theta1 = np.random.rand(1,n)
        self.theta2 = np.random.rand(n,1)
        self.gamma = 0.9 # discount
        self.alpha = 0.2 # learning rate
        self.epsilon = 0.3 # for epsilon-greedy exploration
        self.n = n
        self.empty_sqrs = empty_sqrs
        self.beetle_source = beetle_source

    def Q(self, s, a):
        # s is n by n; a is a scalar
        return np.matmul(np.matmul(self.theta1, s), self.theta2) * a
    
    def gradQ_theta1(self, s, a):
        grad = (np.matmul(s, self.theta2)* a).T
        return grad # / np.linalg.norm(grad)
    
    def gradQ_theta2(self, s, a):
        grad = (np.matmul(self.theta1, s) * a).T
        return grad # / np.linalg.norm(grad)

    def choose_action(self, s):
        action = random.randint(0,self.n**2)
        threshold = random.random()
        if threshold > self.epsilon:
            action = np.argmax([self.Q(s,a) for a in range(self.n**2 + 1)])
        return action

    def update(self, s, a, r, s_next):
        self.theta1 = self.theta1 + self.alpha*(
            r + self.gamma * np.max([self.Q(s_next, a_next) for a_next in range(self.n**2 + 1)]) - self.Q(s,a)
        ) * self.gradQ_theta1(s,a)
        self.theta1 = self.theta1 / np.linalg.norm(self.theta1)

        self.theta2 = self.theta2 + self.alpha*(
            r + self.gamma * np.max([self.Q(s_next, a_next) for a_next in range(self.n**2 + 1)]) - self.Q(s,a)
        ) * self.gradQ_theta2(s,a)
        self.theta2 = self.theta2 / np.linalg.norm(self.theta2)

    def Qlearn_step(self):
        s = np.where(self.sim_train.trees > 0, 1, 0)
        a = self.choose_action(s)
        r = self.sim_train.take_action(a)
        r += self.sim_train.simulate_timestep() # should this be before take_action?
        s_next = np.where(self.sim_train.trees > 0, 1, 0)
        self.update(s, a, r, s_next)

    def train(self, num_iters, num_restarts):
        for sim in range(num_restarts):
            self.sim_train = simulator.Simulator(self.n,self.empty_sqrs,self.beetle_source)
            for i in range(num_iters):
                self.Qlearn_step()
    
    def test(self, num_iters):
        r = 0
        self.sim_test = simulator.Simulator(self.n,self.empty_sqrs,self.beetle_source)
        for i in range(num_iters):
            r += self.sim_test.simulate_timestep()
            s = np.where(self.sim_test.trees > 0, 1, 0)
            a = np.argmax([self.Q(s,action) for action in range(self.n**2 + 1)])
            r += self.sim_test.take_action(a)
        return r
    
    def random_test(self, num_iters):
        r = 0
        self.sim_test = simulator.Simulator(self.n,self.empty_sqrs,self.beetle_source)
        for i in range(num_iters):
            r += self.sim_test.simulate_timestep()
            s = np.where(self.sim_test.trees > 0, 1, 0)
            a = random.randint(0, self.n**2)
            r += self.sim_test.take_action(a)
        return r
    
if __name__ == "__main__":
    learn = QLearning(10,[(0,5),(0,2),(2,6),(2,7),(3,2),(4,9),(3,9),(8,7)],(4,4))
    learn.train(1000, 5)
    rand_r = learn.random_test(1000)
    trained_r = learn.test(1000)
    print("Random policy: ", rand_r)
    print("Learned policy: ", trained_r)