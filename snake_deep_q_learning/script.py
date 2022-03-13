from game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

import os
import re
from collections import deque
import numpy as np
import torch
import random

# Reference: https://www.youtube.com/watch?v=--nsd2ZeYvs

# Bellman Equation
# NewQ(s,a) = Q(s,a) + alpha * [R(s,a) + gamma * Qp(sp,ap) - Q(s,a)]
# Q = model * predict(state0)
# Qnew = R + gamma * max(Q(state1))
# loss = (Qnew - Q) ** 2

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if exceeds MAX_MEMORY
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma = min(self.gamma, 0.999))

    def get_state(self, game: SnakeGame):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._is_collision(point_r)) or 
            (dir_l and game._is_collision(point_l)) or 
            (dir_u and game._is_collision(point_u)) or 
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or 
            (dir_d and game._is_collision(point_l)) or 
            (dir_l and game._is_collision(point_u)) or 
            (dir_r and game._is_collision(point_d)),     

            # Danger left
            (dir_d and game._is_collision(point_r)) or 
            (dir_u and game._is_collision(point_l)) or 
            (dir_l and game._is_collision(point_u)) or 
            (dir_r and game._is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=np.int_) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAXMEM is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            minisample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            minisample = self.memory

        states, actions, rewards, next_states, dones = zip(*minisample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame(AI=True)

    while 1:

        # get old state
        state_old = agent.get_state(game)

        # get move based on current state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # store states in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        # train the long memory (experience replay memory)
        if done:

            # reset game's state
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print(f'Game: {agent.number_of_games}, Score: {score}, Record: {record}')

            total_score += score
            mean_score = total_score / agent.number_of_games

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()