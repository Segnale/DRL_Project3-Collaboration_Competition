# pylint: skip-file
from agent import Agent
from unity_env import UnityEnv
from model import Actor, Critic
import numpy as np
from collections import deque
from utilities import Plotting
# from tensorboardX import SummaryWriter

# fill replay buffer with rnd actions
def init_replay_buffer(env, agent, steps):
    state = env.reset()
    for _ in range(steps):
        action =  (np.random.rand(2,2)*2)-1 # btwn -1..1
        next_state, reward, done = env.step(action)
        agent.replay_buffer.add(state.reshape(-1), action.reshape(-1), np.max(reward), next_state.reshape(-1), np.max(done))
        state = next_state
        if done.any():
            state = env.reset()

def train(env, agent, episodes, steps):
    scores = deque(maxlen=100)
    PScores = []

    # Score Trend Initializaiton
    plot = Plotting(
        title ='Learning Process',
        y_label = 'Score',
        x_label = 'Episode #',
        x_range = 250,
    )
    plot.show()

    # Progress bar o terminal
    import progressbar as pb
    widget = ['training loop: ', pb.Percentage(), ' ',
              pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    # writer = SummaryWriter()
    last_saved = 0
    for episode in range(episodes):
        agent.reset_episode()
        state = env.reset()
        score = np.zeros(env.num_agents)
        for step_i in range(steps):
            action = agent.act(state.reshape(-1))
            next_state, reward, done = env.step(action.reshape(2,-1))
            score += reward
            # step agent
            # if one is done both are done 
            agent.step(state.reshape(-1), action.reshape(-1), np.max(reward), next_state.reshape(-1), np.max(done))
            state = next_state
            if done.any():
                break
        # log episode results
        scores.append(score.max())
        mean = np.sum(scores)/100
        if (episode+1)%50 ==0 :
            print("Episode: {0:d}, Score: {1:f}".format(episode+1,mean))
        timer.update(episode+1)
        #summary = f'Episode: {episode+1}/{episodes}, Steps: {agent.it:d}, Noise: {agent.noise_scale:.2f}, Score Agt. #1: {score[0]:.2f}, Score Agt. #2: {score[1]:.2f}'
        PScores.append(mean)
        plot.Update(list(range(episode+1)),PScores)
        #summary += f', Score: {mean:.3f}'
        # writer.add_scalar('data/score', mean, ep_i)
        if mean > 0.50 and mean > last_saved:
            last_saved = mean
            agent.save('saved/trained_model.ckpt')
        
    timer.finish()

    # Save Training Trend
    end_plot = Plotting(
        title ='Learning Process',
        y_label = 'Score',
        x_label = 'Episode #',
        x_values = list(range(episode-(100-2))),
        y_values = PScores
    )
    end_plot.save('Results/Training.png')

if __name__ == '__main__':
    # hyperparameters
    episodes = 2000
    steps = 2000

    # environment 
    env = UnityEnv(no_graphics=False)
    state_size = env.state_size*2
    action_size = env.action_size*2

    # agent
    agent = Agent(
        state_size, action_size, Actor, Critic,
        lrate_critic=1e-3,
        lrate_actor=1e-4,
        tau=0.01,
        buffer_size=1e6,
        batch_size=256,
        gamma=0.99,
        exploration_mu=0.0,
        exploration_theta=0.15,
        exploration_sigma=0.20,
        seed=np.random.randint(1000),
        update_every=1, 
        update_repeat=1,
        weight_decay=0, 
        noise_decay=0.99995
    )

    # start with rnd actions
    init_replay_buffer(env, agent, int(1e4))

    train(env, agent, episodes, steps)