from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from tensorflow.keras.models import load_model
from helper_funcs import image_compressor, model_input

path = '/'
model = load_model(path + 'model2')
env = gym.make('Breakout-v0')
obss = deque(maxlen=2)
obs = env.reset()
total_score = 0

for i in range(10):
    obss.append(image_compressor(obs))
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    obss.append(image_compressor(obs))

img = plt.imshow(env.render(mode='rgb_array'))
done = False
while not done:
    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = np.argmax(model.predict(model_input(obss[0], obss[1], 4)))
    obs, score, done, _ = env.step(action)
    total_score += score