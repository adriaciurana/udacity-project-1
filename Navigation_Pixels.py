
# coding: utf-8

# # Navigation from pixels
# ---
# 
# In this notebook, the navigation project is presented. 
# 
# The proposed problem is based on getting an agent to learn to determine which bananas are good and which bad.
# For this we have a 84x84 rgb image (state), our task will be to find a function that maps the state in the best possible action.
# 
# To carry out this mapping, it is proposed to use a pretrained mobilenet_v2 (with imagenet).

# All the code to facilitate the organization is in the **code** folder. Where we found:
# * **agent.py**: The proposed agent to solve the navigation problem.
# * **config.py**: Configuration of the hyperparameters of the learning process.
# * **models.py**: The proposed neural networks architectures.

# In[1]:


# Much faster than opencv imread
#get_ipython().system(u'pip install lycon')


# ### Project dependences

# In[1]:


# Import all dependences
import os
import glob
import torch
import lycon
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

# Unity env
from unityagents import UnityEnvironment

# Project dependences
from code.config import Config
Config.BATCH_SIZE = 256
from code.agent import Agent
from code.model import PixelBananasNet


# ### Init the Unity environment

# In[2]:


# Get the unity environment
env = UnityEnvironment(file_name=Config.BANANA_PIXELS_ENV_PATH)

# Select the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]


# ### Display information about the problem:

# In[3]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.visual_observations[0]
print('States look like:')
#plt.imshow(np.squeeze(state))
#plt.show()
state_size = state.shape
print('States have shape:', state.shape)


# ### Init the Agent
# The proposed agent is prepared to allow as input a pretrained mobilenet_v2 architecture.
# 
# The network proposed for this exercise is called **PixelBananasNet**, also with funny name but MORE powerful than the previour BananasNet it will allow us to navigate autonomously using rgb 84x84 camera through the scene and get our precious high resolution bananas! Mmmmm ...

# In[4]:


agent = Agent(PixelBananasNet, state_size[1], action_size, alpha=.5)

# Some problems produces learning crashes
last_weights_path = sorted(glob.glob(os.path.join(Config.CHECKPOINT_PIXELS_BANANA_PATH, "*.pth")), key=lambda f: os.path.getmtime(f), reverse=True)[0]
print(last_weights_path)
agent.load(last_weights_path)


# ### Main loop
# In order to monitoring the learning proces, the following function controls the workflow of the agent.

# In[5]:


def normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return np.stack(((image[..., 0] - mean[0]) / std[0],
                     (image[..., 1] - mean[1]) / std[1],
                     (image[..., 2] - mean[2]) / std[2]), axis=-1)


# In[6]:


def dqn(n_episodes=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = normalize(lycon.resize(env_info.visual_observations[0][0], 
                             height=96, 
                             width=96, 
                             interpolation=lycon.Interpolation.NEAREST).astype(np.float32)).transpose(2, 0, 1)
        score = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = normalize(lycon.resize(env_info.visual_observations[0][0], 
                                      height=96, 
                                      width=96, 
                                      interpolation=lycon.Interpolation.NEAREST).astype(np.float32)).transpose(2, 0, 1)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, 
                       np.array([action]), 
                       np.array([reward]), 
                       next_state, 
                       np.array([done]))
            state = next_state
            score += reward
            if done:
                break 
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.save(os.path.join(Config.CHECKPOINT_PIXELS_BANANA_PATH, 'checkpoint_%d.pth') % (i_episode, ))
            
    return scores


# In[ ]:


# ¡Execute please! ¡I want bananas!
scores = dqn()


# ### Plot learning curve

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ### Play with the last checkpoint
# ![Banana image](doc/banana.jpg)

# In[ ]:


last_weights_path = sorted(glob.glob(os.path.join(Config.CHECKPOINT_PIXELS_BANANA_PATH, "*.pth")), key=lambda f: os.path.getmtime(f), reverse=True)[0]
agent.load(last_weights_path)

env_info = env.reset(train_mode=False)[brain_name]
state = normalize(lycon.resize(env_info.visual_observations[0][0], 
                     height=96, 
                     width=96, 
                     interpolation=lycon.Interpolation.NEAREST).astype(np.float32)).transpose(2, 0, 1)
score = 0
while True:
    action = agent.act(state, eps)
    env_info = env.step(action)[brain_name]
    next_state = normalize(lycon.resize(env_info.visual_observations[0][0], 
                              height=96, 
                              width=96, 
                              interpolation=lycon.Interpolation.NEAREST).astype(np.float32)).transpose(2, 0, 1)
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward
    state = next_state
    if done:
        break
    
print("Score: {}".format(score))

