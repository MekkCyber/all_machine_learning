import numpy as np
import PIL.Image
import tensorflow as tf 
import gym
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

env = gym.make('LunarLander-v2')
env.reset()
#PIL.Image.fromarray(env.render())
state_size = env.observation_space.shape
num_actions = env.action_space.n

q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64,input_shape=state_size,activation="relu"),
    tf.keras.layers.Dense(units=64,activation="relu"),
    tf.keras.layers.Dense(units=num_actions,activation="linear")
])
target_q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64,input_shape=state_size,activation="relu"),
    tf.keras.layers.Dense(units=64,activation="relu"),
    tf.keras.layers.Dense(units=num_actions,activation="linear")
])

optimizer = tf.keras.optimizers.Adam(ALPHA)

experience = namedtuple('Experience',field_names=["state","action","reward","next_state","done"])

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)

def get_action(q_values,epsilon) : 
    if random.random() > epsilon : 
        return np.argmax(q_values.numpy()[0])
    else : 
        return random.choice(np.arange(4))

def compute_loss(experiences,gamma) : 

    states, actions, rewards, next_states, done_vals = experiences
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values,tf.stack([tf.range(q_values.shape[0]),tf.cast(actions,tf.int32)],axis=1))
    q_hat_values = tf.reduce_max(target_q_network(next_states),axis=1)
    loss = tf.reduce_sum(tf.square(q_values-rewards-(gamma*q_hat_values)*(1-done_vals)))
    return loss

def get_new_eps(epsilon) :
    return max(EPS_MIN,EPS_DECAY*epsilon)

def agent_learn(experiences,gamma) : 
    with tf.GradientTape() as tape : 
        loss = compute_loss(experiences,gamma)
    gradients = tape.gradient(loss,q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients,q_network.trainable_variables))
    for target_weights, q_net_weights in zip(target_q_network.weights,q_network.weights) : 
        target_weights.assign(TAU*q_net_weights+(1-TAU)*target_weights)

target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes) : 
    state = env.reset()[0]
    total_points = 0
    for t in range(timestamps) :
        state_q = np.expand_dims(state,axis=0)
        q_values = q_network(state_q)
        action = get_action(q_values,epsilon)
        next_state, reward, done, _, _ = env.step(action)
        memory_buffer.append(experience(state,action,reward,next_state,done))

        if (t+1)%NUM_STEPS_UPDATE==0 and len(memory_buffer)>MINIBATCH_SIZE : 
            experiences = get_experiences(memory_buffer)             
            agent_learn(experiences,GAMMA)
        if done : 
            break
        state = next_state.copy()
        total_points+=reward
    total_points_history.append(total_points)
    epsilon = get_new_eps(epsilon)
    print(f"episode {i} finished, with total points {total_points}")

def plot_history(point_history):

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    points = point_history[lower_limit:upper_limit]

    episode_num = [x for x in range(lower_limit, upper_limit)]

    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    plt.plot(episode_num, points, linewidth=1, color="cyan")
    plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()

plot_history(total_points_history)