import gym

import random
import numpy as np
from collections import deque

import keras
from keras.optimizers import SGD, Adam
from keras.objectives import mean_squared_error
from keras.layers import Dense, InputLayer


"""
An implementation of Q-learning for the cartpole environment.
Based on this gist https://gist.github.com/isseu/7c295d4d2b46e5d9a18dd845ef07dcb9
"""


# Q-learning parameters
#==========================================================================

# size of replay memory
MAX_REPLAY_STATES = 1000

# number of random samples from replay memory to train on each iteration
BATCH_SIZE = 32

# number of games to play
NUM_GAMES_TRAIN = 200

# number of environment observations that constitue a "state"
OBSERVATIONS_PER_STATE = 1

# future discount factor
GAMMA = 0.99

# initial exploration factor
EPSILON_0 = 1.0

# epsilon goes from EPSILON_0 -> EPSILON_MIN over NUM_GAMES_TRAIN
EPSILON_MIN = 0.1


# environment specific variables
#==========================================================================

# instanciate the environment
env = gym.make('CartPole-v1')

OBSERVATION_SIZE = 4
STATE_SIZE = OBSERVATION_SIZE * OBSERVATIONS_PER_STATE

ACTION_CHOICES = np.array([0,1])
ACTION_SIZE = len(ACTION_CHOICES)

INPUT_SIZE = STATE_SIZE
OUTPUT_SIZE = ACTION_SIZE


# create neural network that approximates Q-function
#  this network approximates a function that takes in a state and
#  outputs the discounted future reward for every possible action
#==========================================================================

optimizer = Adam(lr=0.001)
loss = mean_squared_error

model = keras.models.Sequential()
model.add(InputLayer(input_shape=(INPUT_SIZE,)))
model.add(Dense(output_dim=8, activation='tanh'))
model.add(Dense(output_dim=16, activation='tanh'))
model.add(Dense(output_dim=OUTPUT_SIZE, activation='linear'))
model.compile(optimizer=optimizer, loss=loss)


# loop over games and learn!
#==========================================================================

replay = deque()
reward_history = []
epsilon = EPSILON_0

for igame in range(NUM_GAMES_TRAIN):

    # get initial observation
    new_observation = env.reset()

    # create initial state by copying first observation
    new_state = np.concatenate([
        new_observation for i in range(OBSERVATIONS_PER_STATE)])

    loss = 0
    istep = 0
    done = False
    reward_for_game = 0


    # play this game until we're done
    while not done:

        istep += 1
        env.render()

        # select a new action
        #---------------------------------------------------------------
        if random.random() < epsilon:
            action = random.choice(ACTION_CHOICES)
        else:
            # note the input needs to have shape (batch_size, state_size) and
            # the output will have (batch_size, action_size)
            nn_output = model.predict(new_state[np.newaxis, :])
            q = nn_output[0,:]
            action = q.argmax()


        # create new state and update total reward for game
        # here we remove the oldest observation from the state and add
        # a new observation to the end of the state array.
        #---------------------------------------------------------------
        old_state = new_state
        new_observation, reward, done, info = env.step(action)
        new_state = np.roll(new_state, -OBSERVATION_SIZE)
        new_state[-OBSERVATION_SIZE:] = new_observation
        reward_for_game += reward


        # add transition to replay memory <s,a,r,s'> and remove oldest
        replay.append([new_state, reward, action, done, old_state])
        if len(replay) > MAX_REPLAY_STATES:
            replay.popleft()


        # sample from replay memory
        len_mini_batch = min(len(replay), BATCH_SIZE)
        mini_batch = random.sample(replay, len_mini_batch)


        # build training set
        X_train = np.zeros((len_mini_batch, INPUT_SIZE), dtype=np.float32)
        Y_train = np.zeros((len_mini_batch, OUTPUT_SIZE), dtype=np.float32)


        # create targets (Y_train) for network
        for ii in range(len_mini_batch):

            new_bat_state, reward_bat, action_bat, done_bat, old_bat_state = mini_batch[ii]
            old_q = model.predict(old_bat_state[np.newaxis,:])[0]
            new_q = model.predict(new_bat_state[np.newaxis,:])[0]
            update_target = np.copy(old_q)
            # at this point Y_train will be equal to predictions
            if done_bat:
                # if new_state is an absorbing (final) state
                update_target[action_bat] = reward_bat
            else:
                update_target[action_bat] = reward_bat + (GAMMA * np.max(new_q))
            X_train[ii,:] = old_bat_state
            Y_train[ii,:] = update_target

        loss += model.train_on_batch(X_train, Y_train)


    print("[+] End Game {} | Reward {} | Epsilon {:.4f} | TrainPerGame {} | Loss {:.4f} ".format(
        igame, reward_for_game, epsilon, istep, loss / istep))
    reward_history.append(reward_for_game)

    if epsilon >= EPSILON_MIN:
        epsilon -= (1 / (NUM_GAMES_TRAIN))
