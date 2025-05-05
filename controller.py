# %% [markdown]
# # **World Models - Controller (C)**
# 
# 
# ## **Introduction**
# 
# This notebook implements the Controller (C) component of the World Models architecture. In the World Model framework, the agent consists of three components:
# * **Vision (V)**: A Variational Autoencoder (VAE) that compresses visual input into a latent representation
# * **Memory (M)**: A recurrent network (LSTM-MDN) that predicts future states
# * **Controller (C)**: A neural network that takes latent state and hidden state to output actions
# 
# The Controller is trained using an evolutionary strategy (CMA-ES) to optimize performance in the CarRacing environment.
# 
# 
# <img src="imgs/controller.png" width="600" alt="World Models Architecture Diagram">
# 

# %% [markdown]
# ### **Loading Pre-trained Vision (VAE) and Memory (MDN-LSTM) Models**
# 
# First, we load the pre-trained VAE and LSTM-MDN models that form the Vision and Memory components of our World Model.

# %%
import torch
from models.vae import VAE
from models.mdnrnn import MDNRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LATENT_DIM = 32
ACTION_DIM = 3
HIDDEN_DIM = 256
NUM_GAUSSIANS = 5

# VAE model
vision = VAE(3, LATENT_DIM).to(device)
vision.load_state_dict(torch.load('checkpoints/vae.pth', weights_only=False))
vision.eval()

# LSTM-MDN model
memory = MDNRNN(latent_dim=LATENT_DIM, action_dim=3, hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS).to(device)
memory.load_state_dict(torch.load('checkpoints/memory.pth', weights_only=False))
memory.eval()

# %% [markdown]
# ## **Controller Architecture**
# ---
# 
# The Controller receives the latent vector `z` from the VAE and the hidden state `h` from the LSTM-MDN model, and outputs the optimal action to take.
# 
# In the CarRacing environment, the controller produces **three continuous actions**:
# - **Steering**: Range [-1, 1] (using tanh activation)
# - **Acceleration**: Range [0, 1] (using sigmoid activation)
# - **Brake**: Range [0, 1] (using sigmoid activation, limited to 0.8 max)

# %%
import torch.nn as nn

class Controller(nn.Module):
    """
    Controller neural network that maps state (z + h) to actions.
    
    Args:
        state_dim (int): Dimension of the input state (latent + hidden state)
        action_dim (int): Dimension of the output action space
    """
    def __init__(self, state_dim, action_dim=3):
        super(Controller, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, action_dim),
        )

    def forward(self, x):
        """
        Forward pass through the controller network.
        
        Args:
            x (torch.Tensor): Input tensor containing latent state + hidden state
            
        Returns:
            torch.Tensor: Action values for steering, acceleration, and brake
        """
        raw_actions = self.model(x)
        
        # Apply appropriate activations to each action dimension
        steering = torch.tanh(raw_actions[:, 0:1])        # [-1, 1]
        gas = torch.sigmoid(raw_actions[:, 1:2])          # [0, 1]
        brake = torch.sigmoid(raw_actions[:, 2:3]) * 0.8  # [0, 0.8]

        # Ensure brake is reduced when accelerating (realistic vehicle behavior)
        brake = brake * (1-gas)
        
        # Combine all actions into one tensor
        actions = torch.cat([steering, gas, brake], dim=1)
    
        return actions

    def get_action(self, state):
        """
        Get action for a given state without gradient computation.
        
        Args:
            state (torch.Tensor): Current state (z + h)
            
        Returns:
            torch.Tensor: Action to take
        """
        with torch.no_grad():  
            action = self.forward(state)
        return action.squeeze()

# %%
controller = Controller(state_dim = LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).to(device)

print(controller)
print(f'Num params: {sum(p.numel() for p in controller.parameters())}')

# %% [markdown]
# ## **CMA-ES: Covariance Matrix Adaptation Evolution Strategy**
# ---
# 
# ### Theoretical Background
# 
# CMA-ES is a powerful evolutionary algorithm designed for challenging non-linear, non-convex optimization problems. It's particularly effective for training neural networks in reinforcement learning settings where gradient-based methods may struggle.
# 
# **Key Principles of CMA-ES:**
# 
# 1. **Initialization**: The algorithm starts with a multivariate normal distribution defined by an initial **mean** (representing the solution) and a **covariance matrix** which controls exploration.
# 
# 2. **Population Generation**: In each iteration, CMA-ES generates a population of candidate solutions by sampling from the current distribution. The scale of variation is controlled by the **sigma** (step size) parameter.
# 
# 3. **Fitness Evaluation**: Each candidate solution is evaluated on the target problem to determine its fitness.
# 
# 4. **Distribution Update**: The algorithm updates:
#    - The mean vector (shifting toward better solutions)
#    - The covariance matrix (adapting the search distribution to favor promising directions)
#    - The step size (controlling the overall scale of exploration)
# 
# ### Application to World Models
# 
# In our implementation, CMA-ES optimizes the **weights of the controller neural network**. The controller maps latent states `z` and hidden states `h` to actions `a`.
# 
# Since the CarRacing environment has randomized tracks, each controller's performance varies depending on the specific track it encounters. To provide a more reliable fitness evaluation, **each controller is evaluated multiple times** (on different tracks), and the average reward is used as the fitness measure.
# 

# %%
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

def make_env(name='CarRacing-v3'):
    """
    Factory function to create wrapped environment.
    
    Args:
        name (str): Gymnasium environment name
        
    Returns:
        callable: Function that creates the specified environment
    """
    def _init():
        env = gym.make(name, render_mode='rgb_array', 
                       lap_complete_percent=1.0,
                       domain_randomize=False, continuous=True)
        return env
    return _init

def create_vector_envs(num_envs):
    """
    Create vectorized environments for parallel execution.
    
    Args:
        num_envs (int): Number of parallel environments
        
    Returns:
        AsyncVectorEnv: Vectorized environment instance
    """
    return AsyncVectorEnv([make_env() for _ in range(num_envs)], 
                          shared_memory=True)

# %% [markdown]
# ## **Batch Processing for Efficient Evaluation**
# ---
# 
# We implement batch processing to efficiently evaluate multiple controller instances in parallel:
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

def encode_obs_batch(obs_batch, size=(96, 96), device='cuda'):
    """
    Preprocess and encode a batch of observations to latent vectors.
    
    Args:
        obs_batch (numpy.ndarray): Batch of observations from environments
        size (tuple): Target size for image preprocessing
        device (str): Device for computation ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Batch of latent vectors
    """
    # Convert to tensor and normalize
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2) / 255.0
    
    # Crop and resize images
    obs_tensor = obs_tensor[:, :, :-12, :]  # Remove bottom status bar
    obs_tensor = F.interpolate(obs_tensor, size=size, mode='bicubic')
    
    # Encode to latent space
    with torch.no_grad():
        mu, logvar = vision.encode(obs_tensor)
        z = vision.reparameterize(mu, logvar)
    return z


def decode_obs(z):
    """
    Decode latent vector to reconstructed observation.
    
    Args:
        z (torch.Tensor): Latent vector
        
    Returns:
        numpy.ndarray: Reconstructed image as numpy array
    """
    with torch.no_grad():
        obs_recon = vision.decode(z)
    return obs_recon.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

# %%
def process_actions(controllers, x):
    """
    Process actions for all controllers at once in one generation.
    
    Args:
        controllers (list): List of controller models
        x (torch.Tensor): Batch of input states
        
    Returns:
        torch.Tensor: Batch of actions
    """
    return torch.stack([controllers[i].get_action(x[i:i+1]) 
                        for i in range(x.size(0))], dim=0)

def load_weights(controller_class, solutions):
    """
    Load weights (CMA-ES solutions) for each controller in a generation.
    
    Args:
        controller_class: Controller class to instantiate
        solutions (list): List of parameter vectors for controllers
        
    Returns:
        list: List of controller instances with loaded weights
    """
    controllers = []
    with torch.no_grad():
        for params in solutions:
            ctrl = controller_class(state_dim=LATENT_DIM + HIDDEN_DIM, 
                                    action_dim=ACTION_DIM).to(device)
            torch.nn.utils.vector_to_parameters(
                torch.tensor(params, dtype=torch.float32).to(device), 
                ctrl.parameters()
            )
            controllers.append(ctrl)
    return controllers

# %%
import numpy as np

def evaluate_policies(solutions, controller_class, max_steps, memory):
    """
    Evaluate multiple policies in parallel using vectorized environments.
    
    Args:
        solutions (list): List of parameter vectors for controllers
        controller_class: Controller class to instantiate
        max_steps (int): Maximum number of steps per episode
        memory: LSTM-MDN memory model
        
    Returns:
        list: Cumulative rewards for each policy
    """
    num_policies = len(solutions)
    
    # Create controllers with respective parameters
    controllers = load_weights(controller_class, solutions)

    # Create vectorized environments
    envs = create_vector_envs(num_envs=num_policies)
    obs, _ = envs.reset()

    # Initialize hidden states for all policies
    hidden = memory.rnn.init_hidden(num_policies, 'cuda')
    
    # Track rewards and completion status
    cumulative_rewards = np.zeros(num_policies)
    dones = np.full(num_policies, False)

    # Episode rollout
    with torch.no_grad():
        for _ in range(max_steps):
            
            if np.all(dones):
                # Stop if all environments are done
                break

            # Encode observations to latent space
            z_batch = encode_obs_batch(obs)
            
            # Combine latent vectors with hidden states
            h = hidden[0].squeeze(0)
            x = torch.cat([z_batch, h], dim=-1)

            # Get actions from controllers
            actions = process_actions(controllers, x)
            
            # Step environments
            obs, rewards, dones_new, _, _ = envs.step(actions.detach().cpu().numpy())
            
            # Update LSTM hidden states
            z_batch = z_batch.unsqueeze(1)
            actions = actions.unsqueeze(1)
            _, hidden = memory.rnn(z_batch, actions, hidden)
            
            # Update rewards and done status
            dones = np.logical_or(dones, dones_new)
            cumulative_rewards += rewards * (~dones)
            
    envs.close()
    return cumulative_rewards.tolist()

# %% [markdown]
# ## **CMA-ES Training Implementation**
# ---
# 
# Each controller was evaluated 16 times to account for track variability.
# 
# Our implementation uses a modified approach:
# - Population size: 16 (matching available CPU cores)
# - Evaluations per controller: 7 (balance between reliability and computation time)
# - Initial sigma: 0.5 (exploration factor)
# - Sigma decay: 0.992 (gradually reducing exploration)
# 

# %%
import cma
import torch 
import time
import pandas as pd
import numpy as np
import os

INITIAL_SIGMA = 0.1
SIGMA_DECAY = 0.992
np.random.seed(101)  # For reproducibility

def train_cma_es(controller_class, memory, max_generations=100, max_steps=1000, 
                popsize=16, checkpoint=10, rollouts=7):
    """
    Train controller using CMA-ES evolutionary strategy.
    
    Args:
        controller_class: Controller class to train
        memory: LSTM-MDN memory model
        max_generations (int): Maximum number of generations to train
        max_steps (int): Maximum steps per episode
        popsize (int): Population size per generation
        checkpoint (int): Save interval for checkpoints
        rollouts (int): Number of evaluations per controller
        
    Returns:
        tuple: (es, metrics, best_solution)
    """
    # Setup metrics tracking
    metrics = {
        'generation': [],
        'best_reward': [],
        'mean_reward': [],
        'worst_reward': []
    }
    
    # Create directory for checkpoints
    os.makedirs('checkpoints/cma', exist_ok=True)

    # Load existing model if continuing training, otherwise start fresh
    controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, 
                          action_dim=ACTION_DIM).to(device)
    initial_params = torch.nn.utils.parameters_to_vector(
        controller.parameters()).detach().cpu().numpy()

    # Initialize CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(initial_params, INITIAL_SIGMA, {'popsize': popsize})
    
    # Main training loop
    for generation in range(141, max_generations+1):
        
        start_time = time.time()
        
        # Generate population for this generation
        solutions = es.ask()  

        # Evaluate each solution multiple times and average
        mean_rewards = []
        for _ in range(rollouts):
            rewards = evaluate_policies(solutions, controller_class, max_steps, memory)
            mean_rewards.append(rewards)
        mean_rewards = np.mean(mean_rewards, axis=0)       

        # Update CMA-ES with rewards (negative because CMA-ES minimizes)
        es.tell(solutions, [-r for r in mean_rewards])
        
        # Calculate and log training statistics
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        log = (f'Generation ({generation}/{max_generations}) | '
               f'Best Reward: {round(np.max(mean_rewards))} | '
               f'Avg Reward: {np.mean(mean_rewards):.2f} | '
               f'Worst: {round(np.min(mean_rewards))} | '
               f'Time: {int(minutes)}:{int(seconds):02d} | '
               f'Sigma: {es.sigma:.4f}')
        print(log)

        # Update metrics
        metrics['generation'].append(generation)
        metrics['best_reward'].append(np.max(mean_rewards))
        metrics['worst_reward'].append(np.min(mean_rewards))
        metrics['mean_reward'].append(np.mean(mean_rewards))
                
        # Save metrics and best controller at checkpoints
        if generation % checkpoint == 0:
            best_controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, 
                                       action_dim=ACTION_DIM).to(device)
            torch.nn.utils.vector_to_parameters(
                torch.tensor(es.result.xbest, dtype=torch.float32).to(device),
                best_controller.parameters())
            torch.save(best_controller.state_dict(), 
                      f'checkpoints/cma/controller_{generation}.pth')
            pd.DataFrame(metrics).to_csv(f"checkpoints/cma/cma_es_metrics.csv")
            print('--Checkpoint: best controller saved')
    
    # Save final model
    best_controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, 
                               action_dim=ACTION_DIM).to(device)
    torch.nn.utils.vector_to_parameters(
        torch.tensor(es.result.xbest, dtype=torch.float32).to(device),
        best_controller.parameters())
    torch.save(best_controller.state_dict(), f'checkpoints/controller.pth')
    return es, metrics, es.result.xbest

# %%
es, metrics, best_solution = train_cma_es(Controller, memory, max_generations=220, max_steps=1000, popsize=16)

# %%
import pandas as pd
import plotly.graph_objects as go
def hex_to_rgba(hex_color, alpha=0.2):
    """HEX to RGBA."""
    hex_color = hex_color.lstrip('#')
    return f'rgba({int(hex_color[0:2],16)}, {int(hex_color[2:4],16)}, {int(hex_color[4:6],16)}, {alpha})'

import plotly.io as pio

def plot_cma_es_results(metrics, colors=['#86D293', '#FFCF96', '#FF8080'], metric_columns=['best_reward', 'mean_reward', 'worst_reward']):
    fig = go.Figure()

    for metric, color in zip(metric_columns, colors):
        fig.add_trace(go.Scatter(
            x=metrics['generation'],
            y=metrics[metric],
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(color=color, width=2.5),
            fill='tozeroy',
            fillcolor=hex_to_rgba(color, 0.2)
        ))

    fig.update_layout(
        height=600,
        width=900,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Helvetica, Arial, sans-serif",size=14,color="#333333"),
        title=dict(text='CMA-ES Training Results',x=0.5,y=0.95,xanchor='center',yanchor='top',font=dict(size=20,color="#333333")),
        legend=dict(title='',title_font_size=16,font_size=14,bgcolor='rgba(255,255,255,0)',bordercolor='rgba(0,0,0,0)',orientation='h',yanchor='bottom',y=1.02,xanchor='center',x=0.5),
        xaxis=dict(showgrid=True,gridcolor='rgba(200,200,200,0.2)',linecolor='rgba(200,200,200,0.5)',linewidth=1,mirror=True,title='Generation'),
        yaxis=dict(showgrid=True,gridcolor='rgba(200,200,200,0.2)',linecolor='rgba(200,200,200,0.5)',linewidth=1,mirror=True,title='Reward')
    )

    fig.update_traces(hovertemplate='%{y} en Generation %{x}<extra></extra>')
    fig.show()

# %%
metrics = pd.read_csv('checkpoints/cma/cma_es_metrics.csv')
plot_cma_es_results(metrics, colors=['#86D293', '#FFCF96', '#FF8080'], metric_columns=['best_reward', 'mean_reward', 'worst_reward'])

# %%
controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).to(device)
controller.load_state_dict(torch.load('checkpoints/controller.pth', weights_only=True))

# %%
import numpy as np
from gymnasium.wrappers import RecordVideo

def render_policy(env_name, controller, mdnrnn, encode_obs_batch):
    """
    Render a policy in the environment.
    
    Args:
        env_name (str): Name of the environment
        controller: Trained controller model
        mdnrnn: LSTM-MDN memory model
        encode_obs_batch: Function to encode observations
    """
    # Create environment with human rendering
    env = gym.make(env_name, render_mode='human', lap_complete_percent=1.0)

    done = False
    cumulative_reward = 0
    obs, _ = env.reset()
    
    # Initialize hidden state
    h = (torch.zeros(1, HIDDEN_DIM).to(device),
         torch.zeros(1, HIDDEN_DIM).to(device))
    step_count = 1
    
    while True:
        # Encode observation to latent space
        z = encode_obs_batch(obs[np.newaxis, ...])

        # Combine latent and hidden state
        x = torch.cat([z, h[0]], dim=-1)
        
        # Get action from controller
        a = controller.get_action(x)
           
        # Step environment
        obs, reward, done, _, _ = env.step(a.detach().cpu().numpy())
        env.render()
        
        # Update LSTM hidden state
        _, h = mdnrnn.rnn(z, a.unsqueeze(0), h=h)
    
        cumulative_reward += reward
        step_count += 1
        
        # End episode on completion or timeout
        if done or step_count >= 1000:
            break
    
    env.close()
    print(f'Reward: {cumulative_reward:.2f} | Steps: {step_count}')
    
# Visualize the trained controller in action
render_policy('CarRacing-v3', controller, memory, encode_obs_batch)

# %% [markdown]
# ## **Final Evaluation**
# ---
# 
# To rigorously evaluate the performance of our trained controller, we run it for 100 episodes and calculate the mean reward.

# %%
import torch 
import numpy as np
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm

def final_evaluation(controller_class, best_solution, memory, 
                     parallel_rollouts=7, max_steps=1000, popsize=16):
    """
    Comprehensive evaluation of the best controller across multiple episodes.
    
    Args:
        controller_class: Controller class to evaluate
        best_solution: Parameter vector of the best controller
        memory: LSTM-MDN memory model
        parallel_rollouts: Number of parallel evaluation batches
        max_steps: Maximum steps per episode
        popsize: Number of parallel environments
        
    Returns:
        list: All rewards from evaluation
    """
    final_rewards = []
    best_controllers = [best_solution for _ in range(popsize)] 
    
    # Execute multiple evaluation rollouts
    for i in tqdm(range(parallel_rollouts)):
        rewards = evaluate_policies(best_controllers, controller_class, 
                                  max_steps, memory)
        final_rewards.append(rewards)

    # Flatten rewards list and calculate statistics
    all_rewards = np.array(final_rewards).flatten()
    print(f"Performance over {popsize*parallel_rollouts} episodes: "
          f"{np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    return all_rewards

# Load the best controller
controller = Controller(LATENT_DIM+HIDDEN_DIM, ACTION_DIM).to(device)
controller.load_state_dict(torch.load('checkpoints/controller.pth', weights_only=True))

# Extract controller parameters
controller_best_solution = parameters_to_vector(
    controller.parameters()).cpu().detach().numpy()

# Evaluation parameters
popsize = 16
parallel_rollouts = 8

# Run final evaluation
rewards = final_evaluation(Controller, controller_best_solution, memory, 
                         parallel_rollouts=parallel_rollouts, 
                         max_steps=1000, popsize=popsize)


