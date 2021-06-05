# https://github.com/koulanurag/ma-gym/blob/b3cca3f97bf6d6af8871377e6119e2375dbab755/ma_gym/envs/openai/__init__.py
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import gym
import numpy as np
from collections import deque


class MultiDeque:
    def __init__(self, n_agents, maxlen):
        self.deque_per_agents = [deque([], maxlen=maxlen) for _ in range(n_agents)]
        self.maxlen = maxlen
        self.n_agents = n_agents

    def append_n(self, obs_n):
        for i in range(self.n_agents):
            self.deque_per_agents[i].append(obs_n[i])

    def numpy_n(self):
        stacked_obs_n = []
        for i in range(self.n_agents):
            assert len(self.deque_per_agents[i]) == self.maxlen
            stacked_obs = np.concatenate(list(self.deque_per_agents[i]))
            stacked_obs_n.append(stacked_obs)
        return stacked_obs_n


class ObservationStack(gym.Wrapper):
    """ It's a multi agent wrapper over openai's single agent environments. """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.agents = env.agents
        self.k = k
        self.n_agents = len(env.observation_space)

        self.deque_n = MultiDeque(n_agents=self.n_agents, maxlen=k)
        prev_observation_space_n = env.observation_space
        observation_space_n = []
        for prev_observation_space in prev_observation_space_n:
            prev_shape = prev_observation_space.shape
            assert len(prev_shape) == 1
            shape = (prev_shape[0]* self.k,)
            new_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
            observation_space_n.append(new_obs_space)
        self.observation_space = observation_space_n
        #self.observation_space = MultiAgentObservationSpace([self.env.observation_space])
        
    def reset(self):
        #print("ObservationStack reset")
        obs_n = self.env.reset()
        for _ in range(self.k):
            self.deque_n.append_n(obs_n)
        #returns = self.deque_n.numpy_n() 
        #print("len(returns)", len(returns))
        #print("returns[0].shape", returns[0].shape)
        return self.deque_n.numpy_n() 

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.deque_n.append_n(ob)
        return self.deque_n.numpy_n(), reward, done, info

if __name__ == '__main__':
    from utils.make_env import make_env
    env_id = "fullobs_collect_treasure"
    env_id = "simple_tag"
    env_id = "multi_speaker_listener"
    #"multi_speaker_listener"
    env = make_env(env_id, discrete_action=True)
    print(env)
    n_agents = len(env.action_space)
    action_n = [np.zeros(10) for _ in range(n_agents)]

    obs_n = env.reset()
    assert len(obs_n) == n_agents

    obs_n, reward_n, done_n, _ = env.step(action_n)
    assert len(obs_n) == n_agents
    assert len(reward_n) == n_agents
    assert len(done_n) == n_agents
    assert len(obs_n[0]) == 18

    senv = ObservationStack(env, 3)
    sobs_n = senv.reset()
    assert len(sobs_n) == n_agents

    sobs_n, sreward_n, sdone_n, _ = senv.step(action_n)
    assert len(sobs_n) == n_agents
    assert len(sreward_n) == n_agents
    assert len(sdone_n) == n_agents
    assert len(sobs_n[0]) == 18
    # action space [Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5)]