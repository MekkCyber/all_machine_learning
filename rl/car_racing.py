import numpy as np
import PIL.Image
import tensorflow as tf 
import gymnasium as gym
import random
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt

ALPHA = 0.001
TAU = 1e-3
GAMMA = 0.995
EPS_MIN = 0.01
EPS_DECAY = 0.995
num_episodes = 200
total_points_history = []
timestamps = 1000
epsilon = 1
memory_buffer = []
MINIBATCH_SIZE = 64
NUM_STEPS_UPDATE = 4

env = gym.make("CarRacing-v2", domain_randomize=True)

num_actions = (5,)
state_size = env.observation_space.shape
env.reset()
print(env.step([0.65,2.5,9.3]))

q_network = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8,3,input_shape=state_size,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16,3,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,3,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation="relu"),
    tf.keras.layers.Dense(units=5,activation='linear')
])
q_target_network = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8,3,input_shape=state_size,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16,3,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,3,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation="relu"),
    tf.keras.layers.Dense(units=5,activation='linear')
])
optimizer = tf.keras.optimizers.Adam(ALPHA)
q_target_network.set_weights(q_network.get_weights())

total_points_history = []

def get_action(actions,epsilon) :
    if random.random()>epsilon : 
        return np.argmax(actions.numpy()[0])
    else : 
        return np.random.choice(np.arange(5))

for i in range(num_episodes) : 
    total_points = 0
    state = env.reset()[0]
    for t in range(timestamps) :
        state_q = np.expand_dims(state,axis=0)
        actions = q_network(state)
        action = get_action(actions,epsilon)

        next_state, reward, 