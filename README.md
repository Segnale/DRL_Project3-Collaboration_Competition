# Deep Reinforcement Learning Project #3 - Collaboration and Competition

This repository contains the implementation of a DDPG Algorithm to train a couple of Agent playing the 'tennis' simulation environments from [Unity ML-Agents.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
The project is part of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

---

## Installation

To run this code, you will need to download the prebuild Unity enviroment not provided in the repository. You need to select the enviroment for your OS:
* [x Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [x Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [x Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [x Windows (64-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Place the file in the `DRL_Project3-Collaboration_Competition` Folder and unzip.

Beside the Unity enviroment, Python 3.6 must be available with the Unity ML-Agents [(see this link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) installed, and few more packages (see `enviroment.env`).
 
## Environment 

Two agents compete in the tennis envaironment. Each agent has 2 continuous actions available:
- move forward and backward in the direction of the net.
- jump up.

Each agent observes a state space of 8 dimensions. The observations are hold in sequence of 3.

The Reward function of the Agent can be resumed by:
- -0.01 when a the ball falls in the field of the agent or the agent send the ball out of the boundaries.
- +0.1 when the ball is hit by the agent and sent over the net.
- 0 in all the other cases.

The task is episodic, and it is considered solved if one of the two agents can get an average score of +0.5 over 100 consecutive episodes.
 
## Instructions

Open `Tennis.ipynb` and run the code alongside with the provided instructions.
After importing the required packages, the notebook runs a bunch of steps:
1. Starting the Unity enviroment and setup the default brain to address the agents.
2. Analisys of the State and Action Spaces provided by the Unity enviroment.
3. Random actions in the enviroment to fill the replay buffer.
4. Setup of the DDRL agent and hyperparameters for the train process.
5. Training of the agent.
6. See the performance of the trained agents.

Beside the jupyter notebook, few files are used in the calculations:
- `agent.py` provides the implementation of the DDRL algorithm and various functions of the agent.
- `replay_buffer.py`provides the implementation of a class for the replay buffer used by the agent.
- `noise.py` is an implementation of the Ornstein–Uhlenbeck process.
- `model.py` includes the neural networks of the actor and critic used by the agent.
- `utilities.py` contains few classes to support the visualization of the results.

## Referances

- [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
- [OpenAI baselines](https://github.com/openai/baselines)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [Other usefull repositories](https://github.com/ostamand/tennis)
