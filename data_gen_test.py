from tensorflow.keras.models import load_model
from helper_funcs import model_input
import pandas as pd
import numpy as np
import gym

randomness = False
path = 'C:/Users/INTERN/Downloads/VISHESH/DATA/'
try:
    if not randomness:
        model = load_model('model1')
        print('Model Loaded')
except:
    randomness = True

env = gym.make('Breakout-ram-v0')
np.random.seed(0)
env.seed(0)

cols = ['Obs'+str(i) for i in range(128)]
cols.extend(['episode', 'action', 'reward', 'hits'])

episodes = 100
discount_factor = 0.9
epsilon = .5
memory_df = pd.DataFrame(columns=cols)

for i_episode in range(episodes):
    done = False
    obs = env.reset()
    ep_reward, prev_reward, hits = 0, 0, 0
    for i in range(10):
        action = env.action_space.sample()
        prev_obs, _, _, _ = env.step(action)
    
    while not done:
        if np.random.rand() < epsilon or randomness:
            action = env.action_space.sample()
        else:
            x = model.predict(model_input(prev_obs, 4))
            action = np.argmax(x)
            # break
        obs, reward, done, info = env.step(action)
        hits += reward
        ep_reward = reward*10 + ep_reward*discount_factor - .1
        memory_df.loc[len(memory_df.index)] = np.hstack((prev_obs, [i_episode, action, ep_reward, hits]))
        prev_reward = reward
        prev_obs = obs

memory_df.to_csv('random_data_ram2.csv', index=False)
print(memory_df.shape)