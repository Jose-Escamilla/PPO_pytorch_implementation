import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import numpy as np

class MemoryBuffer:
    '''
    Simple buffer to collect experiences and clear after each update.
    The class has six attributes:
    - actions: a list to store actions taken by the agent
    - states: a list to store states observed by the agent
    - logprobs: a list to store the logarithm of the probabilities of actions taken by the agent
    - rewards: a list to store the rewards received by the agent
    - dones: a list to store whether the episode is done or not
    - state_values: a list to store the estimated values of the states observed by the agent
    The class also has a method named "clear_buffer()" which is used to clear the buffer after each update.
    '''
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
    
    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]

class ActorCritic(nn.Module): # represents an actor-critic network for reinforcement learning
    def __init__(self, state_size, action_size, action_std=0.5, hidden_size=32, low_policy_weights_init=True):
        '''
        the module takes a:
        - state_size: An integer representing the dimensionality of the state space.
        - action_size: An integer representing the dimensionality of the action space.
        - action_std: A float representing the standard deviation of the action distribution. Default value is 0.5.
        - hidden_size: An integer representing the number of neurons in the hidden layers. Default value is 32.
        - low_policy_weights_init: A boolean flag indicating whether to initialize the actor network with low weights. 
        '''
        super().__init__()

        self.actor_fc1 = nn.Linear(state_size, 2*hidden_size)
        self.actor_fc2 = nn.Linear(2*hidden_size, hidden_size)

        self.actor_mu = nn.Linear(hidden_size, action_size)
        self.actor_sigma = nn.Linear(hidden_size, action_size)
        
        
        self.critic_fc1 = nn.Linear(state_size, 2*hidden_size)
        self.critic_fc2 = nn.Linear(2*hidden_size, hidden_size)

        self.critic_value = nn.Linear(hidden_size, 1)

        self.distribution = torch.distributions.Normal                          #  representing the probability distribution used to sample actions.

        self.action_var = torch.full((action_size,), action_std*action_std)     # specifies the variance of the action distribution as a tensor of shape (action_size,).
        
        # Boosts training performance in the beginning
        if low_policy_weights_init:
            '''
            the weights of the actor network's mean layer are initialized to small values (0.01) to improve training performance 
            '''
            with torch.no_grad():
                self.actor_mu.weight.mul_(0.01)

    def forward(self, state):
        '''
        The forward method takes in a state tensor and feeds it through the actor and critic networks to obtain the mean and 
        standard deviation of the action distribution, as well as the estimated state value.
        contains the following tensors:
        - mu: A tensor representing the mean of the action distribution for the current state.
        - sigma: A tensor representing the standard deviation of the action distribution for the current state.
        - state_value: A tensor representing the estimated value of the current state.
        '''
        x = torch.tanh(self.actor_fc1(state))
        x = torch.tanh(self.actor_fc2(x))
        mu = torch.tanh(self.actor_mu(x))
        sigma = F.softplus(self.actor_sigma(x))

        v = torch.tanh(self.critic_fc1(state))
        v = torch.tanh(self.critic_fc2(v))
        state_value = self.critic_value(v)

        return mu, sigma, state_value   

    def act(self, state):
        '''Choose action according to the policy.'''
        '''
        Takes a state tensor as input.
        Calls the forward method to generate the mean and standard deviation of a normal distribution for the action.
        Creates a multivariate normal distribution with the mean and diagonal covariance matrix calculated from the mean and variance tensors.
        Draws a sample action from the distribution and computes the log probability of the action using the log_prob method of the distribution.
        Returns the action tensor and its log probability.
        '''
        action_mu, action_sigma, state_value = self.forward(state)

        action_var = self.action_var.expand_as(action_mu)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()
    
    def evaluateStd(self, state, action):
        '''Evaluate action using learned std value for distribution.'''
        '''
        Takes a state tensor and an action tensor as inputs.
        Calls the forward method to generate the mean and standard deviation of a normal distribution for the action.
        Creates a univariate normal distribution with the mean and standard deviation.
        Computes the log probability of the action using the log_prob method of the distribution.
        Returns the log probability and the state value.
        '''
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu.squeeze(), action_sigma.squeeze())
        log_prob = m.log_prob(action)

        return log_prob, state_value

    def evaluate(self, state, action):
        '''Evaluate action for a given state.'''
        '''
        Takes a state tensor and an action tensor as inputs.
        Calls the forward method to generate the mean of the action distribution.
        Creates a multivariate normal distribution with the mean and diagonal covariance matrix calculated from the mean and variance tensors.
        Computes the log probability of the action using the log_prob method of the distribution.
        Computes the state value.
        Computes the entropy of the distribution.
        Returns the action log probability, the state value, and the distribution entropy.
        '''   
        action_mean, _, state_value = self.forward(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO():
    '''Proximal Policy Optimization algorithm.'''
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_clip=0.2, epochs=80, action_std=0.5):
        '''
        - state_size: the size of the state space.
        - action_size: the size of the action space.
        - lr: the learning rate for the optimizer.
        - gamma: the discount factor for future rewards.
        - epsilon_clip: the clipping parameter for the PPO objective.
        - K_epochs: the number of epochs to use for optimizing the PPO objective.
        - policy: an instance of the ActorCritic class, which is a neural network that takes in states and outputs action probabilities and state values.
        - policy_old: another instance of the ActorCritic class, which is used to store a copy of the policy weights for updating purposes.
        - MseLoss: an instance of the PyTorch MSELoss function used for computing the mean squared error loss.
        - optimizer: the optimizer used for updating the policy network weights.
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma  = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = epochs

        self.policy = ActorCritic(self.state_size, self.action_size, action_std)                               
        self.policy_old = ActorCritic(self.state_size, self.action_size, action_std)                                

        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999))                 
    
    def select_action(self, state):
        '''
        takes a state as input, which is a numpy array, and returns an action sampled from the policy_old network.
        used during the execution phase of the reinforcement learning process, where the agent interacts with the environment by selecting actions and receiving rewards. 
        The select_action method allows the agent to choose an action based on the current state and the previous policy learned during the training phase.
        '''
        state = torch.FloatTensor(state.reshape(1, -1))
        
        return self.policy_old.act(state)

    def update(self, memory):
        '''
        takes a memory input, which is an instance of the MemoryBuffer class that contains the agent's experiences from interacting with the environment. 
        The update method uses these experiences to update the agent's neural network.
        Starts by extracting the states, actions, rewards, dones, and log probabilities from the memory input. 
        It then initializes an empty list called discounted_rewards, which will be used to store the discounted rewards for each time step.
        The discounted_reward variable is initialized to 0 and is used to accumulate the discounted rewards for each time step in reverse order. 
        The method then iterates over the rewards and dones lists in reverse order using the reversed() function, calculates the discounted reward for each time step, and adds it to the discounted_reward variable. 
        The discounted reward is calculated using the formula:
        discounted_reward = reward + gamma * discounted_reward * (1 - done)  ¿no se si estoy en lo correcto con esa formula?
        '''
        states = memory.states
        actions = memory.actions
        rewards = memory.rewards
        dones = memory.dones
        log_probs = memory.logprobs 

        discounted_rewards = []
        discounted_reward = 0 
        for i in reversed(range(len(rewards))):
            '''
            This part of the code computes the discounted rewards for each time step of the collected experiences in memory. The discounted_reward variable is initialized to 0, 
            and for each time step in reverse order (i.e., from the last time step to the first), 
            the discounted reward is computed as the sum of the immediate reward and the discounted reward from the next time step (weighted by the discount factor gamma). 
            If the current time step indicates that the episode has ended (dones[i] == True), the discounted reward is reset to 0. 
            The computed discounted rewards are then normalized by subtracting their mean and dividing by their standard deviation.
            '''
            if dones[i] == True:
                discounted_reward = 0  
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # old_state_values = torch.stack(state_values, 1).detach()
        # advantages = discounted_rewards - old_state_values.detach().squeeze()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        '''
        the states, actions, and log probabilities of the actions are extracted from the collected experiences and converted into tensors. 
        The squeeze() method is used to remove any extra dimensions of size 1, and the detach() method is used to ensure that these tensors are detached from their computation graphs 
        (i.e, they will not be used to compute gradients during the subsequent optimization step).
        '''
        states = torch.squeeze(torch.stack(states), 1).detach()
        actions = torch.squeeze(torch.stack(actions), 1).detach()
        old_log_probs = torch.squeeze(torch.stack(log_probs), 1).detach()

        for epoch in range(self.K_epochs):
            '''
            The method first extracts the states, actions, rewards, dones and log probabilities from the input memory object. 
            Then, it calculates the discounted rewards using the formula for the discounted return. The rewards are discounted in reverse order, starting from the last reward and propagating backwards. 
            The discounted rewards are standardized to have a mean of zero and a standard deviation of one.
            '''
            new_log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            new_log_probs = new_log_probs.squeeze()
            advantages = discounted_rewards - state_values.detach().squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon_clip, max=1+self.epsilon_clip)
            loss = -torch.min(ratios*advantages, ratios_clipped*advantages)+ 0.5*self.MseLoss(state_values, discounted_rewards) - 0.01*dist_entropy
            #print(ratios.dtype, advantages.dtype, ratios_clipped.dtype, state_values.dtype, discounted_rewards.dtype, dist_entropy.dtype)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            '''¿esta parte actualiza la red política antigua para que coincida con la red política nueva? '''
        self.policy_old.load_state_dict(self.policy.state_dict())
