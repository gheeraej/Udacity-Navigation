import sys
import json
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from dqn_agent import Agent


###  Train the agent
def train_dqn(env, agent, save_or_load_path, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0
        while True:
            action = agent.act(state, eps)
            
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]  
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), save_or_load_path)
            break
    return scores


### Test the agent
def test_dqn(env, agent):
    brain_name = env.brain_names[0]
    
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0] 
    score = 0
    for t in range(max_t):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]
        score += reward
        if done:
            break
    print("Score of this episode is: %.2f" % (score))         
                
                
### Launcher function
def launch(app_path, train_or_test, save_or_load_path, hyper_file):
    env = UnityEnvironment(file_name=app_path)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=train_or_test)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
        
    agent = Agent(state_size, action_size, seed=42)
    
    if train_or_test:
        if hyper_file is None:
            scores = train_dqn(env, agent, save_or_load_path)
        else:
            with open(hyper_file) as f:
                variables = json.load(f)
                if len(list(set(variables.keys()) & set(["n_episodes", "eps_start", "eps_end", "eps_decay"]))) != 4:
                    print("Parameters file is not well specified")
                    pass
                else:
                    scores = train_dqn(env, agent, save_or_load_path, variables["n_episodes"], variables["eps_start"], variables["eps_end"], variables["eps_decay"])

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        
    else:
        agent.qnetwork_local.load_state_dict(torch.load(save_or_load_path))
        test_dqn(env, agent)
    
    env.close()


if __name__=="__main__":
    if len(sys.argv)<4:
        print("Argument 1 is mandatory and must be the path of the Banana env")
        print("Argument 2 is mandatory and is 1 for train and 0 for test")
        print("Argument 3 is mandatory and is the path of the file to save weights (in train) or to load weights (in test)")
        pass
    if len(sys.argv)<5:
        print("Argument 4 represents the hyper-parameter file defaut to 'None'")
        launch(sys.argv[1], bool(int(sys.argv[2])), sys.argv[3], None)
    else:
        launch(sys.argv[1], bool(int(sys.argv[2])), sys.argv[3], sys.argv[4])
    