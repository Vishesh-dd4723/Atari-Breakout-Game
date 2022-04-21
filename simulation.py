import gym
import matplotlib.pyplot as plt
from IPython import display

env = gym.make('Breakout-v0')

env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
score = 0
done = False
i = 0
while i < 1000 and not done:
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = env.action_space.sample()
    obs, x, done, _ = env.step(action)
    score += x
    i += 1