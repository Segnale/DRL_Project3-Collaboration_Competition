# Deep Reinforcement Learning Project #3 - Collaboration and Competition

This repository contains the implementation of a DQN Algorithm to train a couple of Agent to play the 'tennis' simulation environments from [Unity ML-Agents.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
This aim to solve the navigation problem proposed in the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program by preventing the ball to fall.

---

## Installation

To run this code, you will need to download the prebuild Unity enviroment not provided in the repository. You need to select the enviroment for your OS:
* [x Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [x Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [x Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [x Windows (64-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Place the file in the DRL_Project#1-Navigation Folder and unzip.

Beside the Unity enviroment, Python 3.6 must be available with the Unity ML-Agents [(see this link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) installed, and few more packages (see `enviroment.env`).
 
## Environment 

Two agents compete in the envaironment. Each agent has only 2 continuous actions available:
- move forward and backward.
- moving up and down.

Each agent observe a state space of 24 dimensions.

The Reward function of the Agent can be resumed by:
- -0.01 when a the ball falls in the field of the agent.
- +0.1 when the ball is hit by the agent.
- 0 when the ball is on fly.

The task is episodic, and it is considered solved if the agent can get an average score of +0.5 over 100 consecutive episodes.
 
## Instructions

Open `Tennis.ipynb` and run the code alongside with the provided instructions.
After importing the required packages, the notebook runs in a bunch of steps:
1. Starting the Unity enviroment and setup the defaul brain to address the agents.
2. Analisys of the State and Action Spaces provided by the Unity enviroment.
3. Random actions in the enviroment to fill the replay buffer (example of the interaction between an agent and the enviroment).
4. Initial setup of the parameters of the DQN Algorithm to train the agent.
5. Training of the agent.
6. See the performance of the trained agents.

The file `my_methods.py` provides the implementation of the DQN algorithm.
The agent class is included in `dqn_agent.py` while the Deep Neural Network models used by the agent are included in the `model.py`.

## ToDo list

Much more stuff can be done to around this project.

### Training Algorithm improvements

This Deep Q-Learning algorithm can be still improved with proved extensions.
Here the planned implementations:
1.Prioritized replay
2.Distributional DQN

### Deep Analysis of the Paramenters and Architecture

Wide analysis of the impact of the training parameters and model architecture on the training performance. For that, I will setup a specific notebook.

### Step to Pixel Based State Space

Use the *Navigation_Pixels.ipynb* and adapt the agent code to solve the banana collection enviroment using raw pixels. That will require mainly the modification of the `model.py` to include convolutional layers.
