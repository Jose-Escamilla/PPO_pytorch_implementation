import gym
import torch
import numpy as np
from collections import deque
import time
import imageio
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO, MemoryBuffer

env_name = "BipedalWalker-v3"                           #  environment that you want to train your agent on.

n_episodes = 1000                                       # number of training episodes
max_steps = 1600                                        # the maximum number of steps that your agent can take within each episode.
update_interval = 4000                                  # number of steps after which you want to update your agent's neural network weights.
log_interval = 20                                       # number of episodes after which you want to print the average score and episode length of your agent's performance.
solving_threshold = 300                                 # minimum score that your agent needs to achieve in order for you to consider it as "solved" the environment.
time_step = 0                                           # current time step of your agent's training (this is initialized to zero).

render = False                                          # whether you want to render the environment during training (visualize your agent's performance).
train = True                                            # whether you want to train your agent or just test its performance (if set to False, your agent will not update its neural network weights).             
pretrained = False                                      # whether you want to use a pretrained agent (if set to True, your agent will load the weights of a previously trained model instead of starting from scratch).
tensorboard_logging = True                              # whether you want to log the training progress using TensorBoard 

env = gym.make(env_name)                                # OpenAI gym environment used
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
state_size = env.observation_space.shape[0]             # number of features in the observation space of your environment 
action_size = env.action_space.shape[0]                 # number of possible actions that your agent can take 

scores = deque(maxlen=log_interval)                     # the scores achieved by your agent during training (you are keeping track of the last log_interval scores).
max_score = -1000                                       # the highest score achieved by your agent during training (this is initialized to a very low value).
episode_lengths = deque(maxlen=log_interval)            # stores the lengths of each episode played by your agent (again, you are keeping track of the last log_interval episode lengths).
rewards =  []                                           # a list that stores the rewards obtained by your agent during training

memory = MemoryBuffer()                                 # a replay memory buffer that stores the experiences (i.e., state-action-next state transitions) encountered by your agent during training (this is used to randomly sample batches of experiences during the PPO algorithm).

agent = PPO(state_size, action_size)                    # the PPO agent that you are training (this is an instance of the PPO class).

if not train:
    '''
    The first line of this code block checks if the train variable is set to False. If it is, it means that you are not training your agent, but rather just testing its performance.
    This is because during testing, you do not want your agent to update its weights, but rather just use its current policy to choose actions based on the observed states.
    If train is set to True, it means that you are training your agent. In that case, you are creating a new SummaryWriter object from the tensorboardX library, 
    which is used to log and visualize the training progress using TensorBoard. Which will create a new log directory with a unique timestamp every time you run your code. 
    '''
    agent.policy_old.eval()
else:
    writer = SummaryWriter(log_dir='logs/'+env_name+'_'+str(time.time()))

if pretrained:
    '''
    The first if statement checks if pretrained is set to True. 
    If it is, it means that you want to load the pretrained weights of your agent's neural network from a previously saved file.
    After loading the pretrained weights (if any), you are creating an imageio writer object to save a GIF animation of the environment during training. 
    '''
    agent.policy_old.load_state_dict(torch.load('./PPO_modeldebug_best_'+env_name+'.pth'))
    agent.policy.load_state_dict(torch.load('./PPO_modeldebug_best_'+env_name+'.pth'))

writerImage = imageio.get_writer('./images/run.gif', mode='I', fps=25)

for n_episode in range(1, n_episodes+1):
    '''
    iterate over the n_episodes specified in the hyperparameters. For each episode, the code resets the environment using env.reset(), 
    which returns the initial observation/state of the environment. This initial state is then converted to a PyTorch tensor using torch.FloatTensor() 
    and reshaped to have a shape of (1, -1), where -1 indicates that the size of the second dimension is inferred from the size of the input array.
    '''
    state = env.reset()
    state = torch.FloatTensor(state.reshape(1, -1))

    episode_length = 0
    for t in range(max_steps):
        '''
        iterates over a maximum of max_steps steps in the environment. For each step, the code increments the time_step counter.
        This method returns two values: the action to take, and the log_prob of that action under the current policy.
        '''
        time_step += 1

        action, log_prob = agent.select_action(state, memory)
        
        '''
        The state is then reshaped to have a shape of (1, -1) and added to the states list of the memory buffer. Similarly, 
        the action and log_prob are added to the actions and logprobs lists of the memory buffer.
        '''
        state = torch.FloatTensor(state.reshape(1, -1))

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)
        '''
        The action tensor is converted to a NumPy array using action.data.numpy(), flattened using .flatten(), and passed to the environment's step() method, 
        which takes a step in the environment using that action. The step() method returns four values: the next state of the environment (state), 
        the reward received for taking the action (reward), whether the episode has ended (done), and additional information (_)
        '''
        state, reward, done, _ = env.step(action.data.numpy().flatten())

        '''
        The reward and done values are then added to the corresponding lists in the memory buffer, and the reward value is appended to the rewards list for logging purposes.
        i nitializes the state_value variable to 0. This variable will be used later in the code to compute the value of the current state under the current value function.
        '''
        memory.rewards.append(reward)
        memory.dones.append(done)
        rewards.append(reward)
        state_value = 0
        
        if render:
            '''
            If the render flag is set to True, the code captures the current frame of the environment 
            '''
            image = env.render(mode = 'rgb_array')
            # if time_step % 2 == 0:
            #     writerImage.append_data(image)

        if train:
            '''
            If the train flag is set to True, the code checks if the current time step is a multiple of update_interval. 
            If it is, the code calls the update() method of the agent object, passing in the memory buffer.
            This method updates the policy and value function of the agent using the data stored in the memory buffer, 
            which represents a batch of experience collected from the environment. After the update is complete, 
            the time_step counter is reset to 0 and the memory buffer is cleared.
            '''
            if time_step % update_interval == 0:
                agent.update(memory)
                time_step = 0
                memory.clear_buffer()
        '''
        After the for loop that steps through the environment is complete, 
        the code appends the length of the current episode (episode_length) to the episode_lengths deque for logging purposes
        '''
        episode_length = t

        if done:
            break
    '''
    It then computes the total reward received in the current episode (total_reward) as the sum of the rewards stored in the memory buffer from the current episode
    '''
    episode_lengths.append(episode_length)
    total_reward = sum(memory.rewards[-episode_length:])
    scores.append(total_reward)
    
    if train:
        '''
        If the train flag is set to True, the code checks if the current episode number is a multiple of log_interval. 
        If it is, the code prints the current episode number, the average episode length (computed using np.mean() over the episode_lengths deque), 
        and the average score (computed using np.mean() over the scores deque). If the average score is greater than the solving_threshold, 
        the code prints a message indicating that the environment has been solved and saves the model parameters to a file using torch.save().
        '''
        if n_episode % log_interval == 0:
            print("Episode: ", n_episode, "\t Avg. episode length: ", np.mean(episode_lengths), "\t Avg. score: ", np.mean(scores))

            if np.mean(scores) > solving_threshold:
                print("Environment solved, saving model")
                torch.save(agent.policy_old.state_dict(), 'PPO_model_solved_{}.pth'.format(env_name))
            
        if total_reward > max_score:
            '''
            The code also checks if the total reward received in the current episode (total_reward) is greater than the max_score seen so far. 
            If it is, the code updates max_score and saves the model parameters to a file using torch.save(). 
            '''
            print("Saving improved model")

            max_score = total_reward
            torch.save(agent.policy_old.state_dict(), 'PPO_modeldebug_best_{}.pth'.format(env_name))

        if tensorboard_logging:
            '''
            If tensorboard_logging is set to True, the code logs the total reward and average score for the current episode, 
            as well as the episode length and average episode length
            '''
            writer.add_scalars('Score', {'Score':total_reward, 'Avg. Score': np.mean(scores)}, n_episode)
            writer.add_scalars('Episode length', {'Episode length':episode_length, 'Avg. Episode length': np.mean(episode_lengths)}, n_episode)
    
    else:
        print("Episode: ", n_episode, "\t Episode length: ", episode_length, "\t Score: ", total_reward)
    '''
    If the train flag is set to False, the code simply prints the current episode number, the episode length, and the total reward received. 
    In this case, the total_reward variable is set to 0 after logging to prepare for the next episode.
    '''    
    total_reward = 0
