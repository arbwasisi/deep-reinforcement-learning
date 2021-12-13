import numpy as np
from ddpg_agent import Agent, ReplayBuffer
import torch

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor

class MADDPG():
    
    def __init__(self, state_size, action_size, num_agents, seed):
        """Learns from centralized training and decentralized execution."""
        
        # Initialize all agents
        self.num_agents = num_agents
        self.agents = [Agent(state_size, action_size, seed) for i in range(num_agents)]
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        self.memory.add(states, actions, rewards, next_states, dones)
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences_dic = self.memory.sample(self.num_agents)
            self.learn(experiences_dic, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions from all agents given their states."""
        
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise)
            actions.append(action)
        return np.vstack([a for a in actions])

    def learn(self, experiences_dic, gamma):
        """Update policy and value parameters using given batch of experience tuples."""

        for i, agent in enumerate(self.agents):
            agent.learn(experiences_dic[i,'states'], experiences_dic[i,'actions'], experiences_dic[i,'rewards'],
                        experiences_dic[i,'next_states'], experiences_dic[i,'dones'], gamma)

    def save_agents(self):
        """Saves best model parameters."""
        
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"checkpoint_actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{i}.pth")