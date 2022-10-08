from tensorflow.keras.models import load_model
from collections import deque
from helper_funcs import *
import pandas as pd
import numpy as np
import gym

randomness = False
path = 'C:/Users/INTERN/Downloads/VISHESH/DATA/'
try:
    if not randomness:
        model = load_model('model2')
except:
    randomness = True

env = gym.make('Breakout-v0')

cols = ['Obs'+str(i)+'_'+str(m)+'_'+str(n) for i in range(2) for m in range(80) for n in range(75)]
cols.extend(['action', 'reward'])

episodes = 100
life_memory = []
memory_df = pd.DataFrame(columns=cols)
i_episode = 0
epsilon = 0
discount_factor = 0.9
stepper = True

while i_episode < episodes:
    episodic_memory = []
    done = False
    obss = deque(maxlen=2)
    obs = env.reset()
    ep_reward = 0
    prev_reward = 0
    for i in range(10):
        obss.append(image_compressor(obs))
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        obss.append(image_compressor(obs))
    
    while not done:
        step_memory = []
        stepper = not stepper
        if np.random.rand() < epsilon or randomness:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(model_input(obss[0], obss[1], 4)))
        obs, reward, done, info = env.step(action)
        ep_reward = (reward - prev_reward)*50 + ep_reward*discount_factor
        if stepper:
            continue
        step_memory.extend(obss[0].flatten())
        step_memory.extend(obss[1].flatten())
        step_memory.extend([action, ep_reward])
        # episodic_memory.append(list_to_pdSeries(keys=cols, values=step_memory))
        memory_df.loc[len(memory_df.index)] = step_memory
        obss.append(image_compressor(obs))
        prev_reward = reward

    # life_memory.extend(episodic_memory)
    i_episode += 1

# memory_df = pd.concat(life_memory, axis=1).T
print(memory_df.shape)

memory_df.to_csv(path + 'random_data.csv', index=False)