# TODO : set N to random and score = awt/n

from os import read
import torch
import random
import time
import numpy as np
from collections import deque
from model import Linear_QNet,QTrainer
from helper import plot
import random
import sys

MAX_MEMORY = 100000
BATCH_SIZE = 20
LR = 0.01

def tanh(x) :
    # if(x>1000) : print("enfcfccdddd")
    # p = np.exp(x)
    # n = np.exp(-x) 
    # return (p-n)/(p+n)
    return np.tanh([x])[0]

def sigmoid(x) :
    return 1/(1+np.exp(-x))

class ready_queue :
    
    def __init__(self) :
        self.N = 0 
        self.pid = [] 
        self.state = {} 
        # total wait, wait since last execution, total execution, execution remaining, priority

class table : 

    def __init__(self) : 
        self.N = 0
        self.pid = []
        self.arrival = []
        self.burst = []
        self.priority = []
        self.i = 0

    def get(self,N) :
        self.N = N
        self.pid = [None]*N
        self.arrival = [None]*N
        self.burst = [None]*N
        self.priority = [None]*N
        for i in range(N) :
            self.pid[i] = i
            r = random.uniform(0.136,1)
            ran = (-np.log(r))//0.2
            if(i==0) : ran = 0
            else : ran = ran + self.arrival[i-1]
            self.arrival[i] = ran
            self.burst[i] = random.randint(1,20)
            self.priority[i] = random.uniform(0,1)

class algos() : 

    def __init__(self,table) : 
        self.table = table

    def fifo(self) : 
        t = 0 
        twt = 0 
        for i in range(self.table.N) : 
            wt = t-self.table.arrival[i]
            t = t + self.table.burst[i]
            if(wt<0) : 
                t = self.table.arrival[i] + self.table.burst[i] 
                wt = 0
            twt = twt + wt
        return twt/self.table.N

class sim : 

    def __init__(self) :
        self.wt = {}

    def step(self,pid,tq) : 
        self.cpu = pid 
        dt = min(tq,self.rq.state[pid][3])
        self.t = self.t + dt 
        self.rq.state[pid][3] = self.rq.state[pid][3] - dt 
        return dt

    def reset(self,n) : 
        self.t = 0
        self.cpu = -1
        self.rq = ready_queue() 
        self.table = table()
        self.table.get(n) 

    def play_step(self,moves) : 

        # Simulating sending process with highest score to cpu

        dt = 0
        if(self.rq.N>0) :
            pid = max(moves,key=moves.get)
            dt = self.step(pid,5) 
        else : 
            self.t = self.t + 1
            if self.table.N == self.table.i : return {},True,np.array(list(self.wt.values())).mean()

        rewards = {} 

        # Getting Rewards & Updating States

        for i in range(self.rq.N) : 
            p = self.rq.pid[i]
            if p!=pid : 
                rewards[p] = -self.rq.state[p][1]/100
                self.rq.state[p][0] = self.rq.state[p][0] + dt
                self.rq.state[p][1] = self.rq.state[p][1] + dt
            else : 
                rewards[p] = self.rq.state[p][-1]
                if(self.rq.state[p][3]==0) : 
                    rewards[p] = rewards[p] + 1
                    self.wt[pid] = self.rq.state[pid][0]
                self.rq.state[p][1] = 0
                self.rq.state[p][2] = self.rq.state[p][2] + dt

        #print(list(rewards.values()))

        # Updating Ready Queue 

        i = self.table.i
        while(i<self.table.N and self.table.arrival[i]<self.t) : 
            self.rq.N = self.rq.N + 1
            p = self.table.pid[i]
            self.rq.pid.append(p)
            self.rq.state[p] = [0,0,0,self.table.burst[i],self.table.priority[i]]
            i = i + 1
        self.table.i = i

        # Returning 

        done = False
        score = 0
        if i==self.table.N and self.rq.N==0 : 
            done = True
            score = np.array(list(self.wt.values())).mean()

        return rewards,done,score


class Agent : 

    def __init__(self) :
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (<1)   
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(5,16,1)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,rq,pid) :

        s = np.array(rq.state[pid])
        for i in range(4) : 
            s[i] = sigmoid(s[i])
        return s

    def remember(self,state,action,reward,next_state,done) :
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self) : 
        if(len(self.memory)>BATCH_SIZE) :
            mini_sample = random.sample(self.memory,BATCH_SIZE) 
        else : 
            mini_sample = self.memory

        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done) :
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state) : 
        self.epsilon = max(2,80 - self.n_games)

        if random.randint(0,200) < self.epsilon : 
            move = random.uniform(0,1)
        else : 
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move = prediction[0].item()
        
        #try : 
            #print(move, prediction)
        #except : print(move)

        return move

def train() :
    plot_scores = []
    plot_mean_scores = []
    plot_fifo = []
    h = 100
    hi = 1
    total_score = 0
    record = 0
    game = sim()
    game.reset(50)
    agent = Agent()
    
    while True : 
        state_olds = {}
        final_moves = {}
        for i in range(game.rq.N) : 
            pid = game.rq.pid[i]
            state_olds[pid] = agent.get_state(game.rq,pid)
            final_moves[pid] = agent.get_action(state_olds[pid])

        rewards,done,score = game.play_step(final_moves)
        
        state_news = {}

        for pid in rewards.keys() : 

            state_news[pid] = agent.get_state(game.rq,pid)

            if(game.rq.state[pid][3]==0) : 
                del game.rq.state[pid]
                del game.rq.pid[game.rq.pid.index(pid)]
                game.rq.N = game.rq.N - 1

            agent.train_short_memory(state_olds[pid],final_moves[pid],rewards[pid],state_news[pid],done)

            agent.remember(state_olds[pid],final_moves[pid],rewards[pid],state_news[pid],done)

        if done : 
            alg = algos(game.table)
            ffo = alg.fifo()
            game.reset(50)
            agent.n_games +=1
            agent.train_long_memory()

            # if score > record :
            #     record = score
            #     agent.model.save()

            agent.model.save()

            print('Game',agent.n_games,'Score',score,'Record',record)

            plot_scores.append(score)
            plot_fifo.append(ffo)
            total_score+=score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            hi = hi + 1
            if(hi==h) : plot(plot_scores,plot_fifo)


if __name__=='__main__' :
    train()