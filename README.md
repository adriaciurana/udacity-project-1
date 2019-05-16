# Project 1: Navigation

### Introduction

In this project it is proposed to use an autonomous agent for the capture of wild bananas. Unfortunately, there are two types of bananas: yellow (perfect for consumption), blue (very toxic).Our objective is to try to capture the *"good"* bananas while avoiding the *“bad”* ones.

![Banana Project](doc/banana.jpg)

In the first instance, we should try to formalize the problem proposed in one of RL.

| Variable name | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| **state**     | A set of 37 sensors that describe the information of the current state. |
| **actions**   | A set of 4 possible actions: move forward (0), move backward (1), turn left (2), turn right (3). |
| **reward**    | +1 get a good banana, -1 get a bad one.                      |

Once the problem is defined, we must remember that an RL problem is simply a **maximization problem.** Where we want to maximize the final reward based on all the actions taken. In addition, the problem we face is a problem in a **episodic** and **continuous state-space.**  



### Project structure

The project has estructured as following:

* **README.md**
* **Report.pdf**: Document that explain all implementation.
* **Navigation.ipynb:** The navigation project jupyter notebook.
* **Navigation_Pixels.ipynb**: The navigation pixels project jupyter notebook.
* *code*: implementation of the DQN, agent, etc.
  * **agent.py**: Definition of the used agent.
  * **config.py**: Configuration of paths, pytorch device and hyperparameters.
  * **model.py**: Network architectures.
  * *mobilenetv2*: folder of the mobilenetv2 implementation.
* *doc*: additional images, etc for readme.
* *envs*: location of the used environments.
  * *Banana_Linux*
  * *VisualBanana_Linux*
* *checkpoints*: saved checkpoints during training:
  * *banana*
  * *banana_pixels*

### Getting Started

1. Download the environment and put in the envs location.
2. Change the path ```code/config.py```  if your operative system is different than Linux.
3. Use ```jupyter notebook``` to view both projects.
