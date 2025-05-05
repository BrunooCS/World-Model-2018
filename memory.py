# %% [markdown]
# # **World Model - Memory *(M)*** 
# 
# This notebook implements the Memory component of a World Model framework. The Memory module uses a combination of recurrent neural networks and mixture density networks to predict future states based on current observations and actions.
# 
# ### **Workflow Overview**
# - **Data Generation and Processing:** Collect state-action sequences from environment interactions
# - **Latent Vector Extraction:** Convert raw observations into compact latent representations (z)
# - **Dataset Creation:** Prepare sequential data for MDN-RNN training
# - **MDN-RNN Model Implementation:** Build and train the predictive memory model
# - **Visualization and Evaluation:** Analyze model performance through reconstructions
#    
# 

# %% [markdown]
# ### **1. Data Preparation: CarRacingDataset**
# ---
# The `CarRacingDataset` class creates a PyTorch dataset from pre-recorded environment interactions stored in HDF5 format.
# 
# #### **Key Features**
# 
# - **Image Transformation:** Uses a pre-trained Variational Autoencoder (VAE) to encode raw images into latent vectors (z)
# - **Sequence Generation:** Prepares temporal sequences for LSTM training
# - **Data Structure:**
#   - **z_t**: Current state in latent space
#   - **z_{t+1}**: Next state (prediction target)
#   - **action_t**: Action taken at time t

# %%
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CarRacingDataset(Dataset):
    def __init__(self, h5_path='car_racing_data.h5', transform=None, vae=None, device=None):
        self.h5_path = h5_path
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.vae = vae
        self.device = device

        # Open HDF5 to get the dimensions of the dataset
        with h5py.File(self.h5_path, 'r') as h5f:
            self.num_episodes = h5f['images'].shape[0]
            self.max_steps = h5f['images'].shape[1]

        self.h5_file = None 
        self.vae.eval() 

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        """
        Retrieves the episode corresponding to the given index.
        
        Args:
            idx (int): index of the episode to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'z': Shape tensor (episode_length, z_dim).
                - 'z_next': Shape tensor (episode_length, z_dim)
                - 'action': Shape tensor (episode_length, action_dim)
                - 'reward': Shape tensor (episode_length,)
                - 'done': Shape tensor (episode_length,)
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        images = self.h5_file['images'][idx]       
        actions = self.h5_file['actions'][idx]     
        rewards = self.h5_file['rewards'][idx]     
        dones = self.h5_file['dones'][idx]        

        # Process each sequence to get the encoded images
        with torch.no_grad():
            img_tensors = torch.stack([self.transform(img) for img in images])  
            img_tensors = img_tensors.to(self.device)
            mu, logvar = self.vae.encode(img_tensors)
            z = self.vae.reparameterize(mu, logvar)  
            
        # Obtain the next images z_(t +1)
        z_next = z[1:]  
        z = z[:-1]      

        actions = torch.tensor(actions[:-1], dtype=torch.float32)
        rewards = torch.tensor(rewards[:-1], dtype=torch.float32)
        dones =  torch.tensor(dones[:-1], dtype=torch.float32)


        return z, z_next, actions, rewards, dones


    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

# %% [markdown]
# ## **2. RNN-MDN Model Architecture**
# ---
# 
# The MDN-RNN model combines two components:
# 1. A **Long Short-Term Memory (LSTM)** network that captures temporal dependencies
# 2. A **Mixture Density Network (MDN)** that outputs probability distributions rather than point estimates
# 
# ### Model Function
# 
# The MDN-RNN predicts the distribution of the next latent state given the current state, action, and hidden state:
# 
# $$P(z_{t+1} \mid a_t, z_t, h_t)$$
# 
# <img src="imgs/rnn_mdn.png" width="600">
# 
# ### Component Details
# 
# #### LSTM (Long Short-Term Memory)
# - Functions as the memory component, maintaining information about past states and actions
# - Processes sequences of latent vectors and actions
# - Updates its hidden state to capture temporal dependencies
# 
# #### MDN (Mixture Density Network)
# - Predicts a mixture of Gaussian distributions rather than a single output value
# - Enables the model to capture uncertainty and multimodal predictions
# - Outputs three parameter sets:
#   - **π (pi)**: Mixture weights (probabilities summing to 1)
#   - **μ (mu)**: Means of Gaussian components
#   - **σ (sigma)**: Standard deviations of Gaussian components
# 
# <img src="imgs/rnn_mdn_2.png" alt="picture mdn" width="800"/>
# 
# ### Loss Function: Negative Log-Likelihood (NLL)
# 
# For a mixture of Gaussians, the probability of a value x is:
# 
# $$P(x|\mu_k, \sigma_k) = \frac{1}{\sqrt{2 \pi \sigma_k^2}} \exp\left( -\frac{(x - \mu_k)^2}{2 \sigma_k^2} \right)$$
# 
# The negative log-likelihood loss is:
# 
# $$\text{Loss} = -\log \left( \sum_{k=1}^{M} \pi_k \cdot \text{P}(x|\mu_k, \sigma_k) \right)$$
# 
# ### Numerical Stability: Log-Sum-Exp Trick
# 
# To prevent underflow issues when computing probabilities of the mixture components, we use the log-sum-exp trick:
# 
# $$\mathcal{L} = -\log \left( \sum_{k=1}^{M} \pi_k \cdot \text{P}(x|\mu_k, \sigma_k) \right) = -\log \left( \sum_{k=1}^{M} \exp \left( \log (\pi_k) + \log (P(x \mid \mu_k, \sigma_k)) \right) \right)$$
# 
# With the trick applied:
# 
# $$\mathcal{L} = - \left( \max_k \log(a_k) + \log \left( \sum_{k=1}^{M} \exp(\log(a_k) - \max_k \log(a_k)) \right) \right)$$
# 
# where $a_k = \pi_k \cdot \text{P}(x|\mu_k, \sigma_k)$
# 
# PyTorch provides an optimized implementation via `torch.logsumexp()`.

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RNN(nn.Module):
    """
    Recurrent neural network that processes latent states and actions.
    
    Args:
        latent_dim: Dimension of the latent state vector
        action_dim: Dimension of the action vector
        hidden_dim: Dimension of the hidden state
        num_layers: Number of LSTM layers
    """
    def __init__(self, latent_dim, action_dim, hidden_dim, num_layers=1):
        super(RNN, self).__init__()
        
        self.lstm = nn.LSTM(input_size=latent_dim + action_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, z, a, h=None):
        """
        Forward pass through the RNN.
        
        Args:
            z: Latent state tensor [batch_size, seq_len, latent_dim]
            a: Action tensor [batch_size, seq_len, action_dim]
            h: Hidden state tuple (h_n, c_n) or None for zero initialization
            
        Returns:
            outs_rnn: Output tensor [batch_size, seq_len, hidden_dim]
            h: Updated hidden state tuple (h_n, c_n)
        """
        x = torch.cat([z, a], dim=-1)
        outs_rnn, h = self.lstm(x, h)
        return outs_rnn, h
    
    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state with zeros"""
        return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                torch.zeros(1, batch_size, self.hidden_dim).to(device))
    
    
class MDN(nn.Module):
    """
    Mixture Density Network that produces parameters for a mixture of Gaussians.
    
    Args:
        latent_dim: Dimension of the latent state vector
        hidden_dim: Dimension of the hidden state from RNN
        num_gaussians: Number of Gaussian components in the mixture
    """
    def __init__(self, latent_dim, hidden_dim, num_gaussians):
        super(MDN, self).__init__()
        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim
        
        # Network layers for mixture parameters
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)          # Mixture weights (π)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)   # Means (μ)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim) # Std devs (σ)
        
    def forward(self, outs_rnn, tau):
        """
        Forward pass through the MDN.
        
        Args:
            outs_rnn: RNN output tensor [batch_size, seq_len, hidden_dim]
            tau: Temperature parameter for controlling prediction sharpness
            
        Returns:
            pi: Mixture weights [batch_size, seq_len, num_gaussians]
            mu: Gaussian means [batch_size, seq_len, num_gaussians, latent_dim]
            sigma: Gaussian std devs [batch_size, seq_len, num_gaussians, latent_dim]
        """
        tau = torch.tensor(tau).to(outs_rnn.device)
        
        pi = self.fc_pi(outs_rnn)
        mu = self.fc_mu(outs_rnn)
        sigma = self.fc_sigma(outs_rnn)
        
        # Reshape outputs to [batch_size, seq_len, num_gaussians, latent_dim]
        batch_size, seq_length, _ = outs_rnn.size()
        mu = mu.view(batch_size, seq_length, self.num_gaussians, self.latent_dim)
        sigma = sigma.view(batch_size, seq_length, self.num_gaussians, self.latent_dim)
        
        # Ensure sigma is positive and apply temperature scaling
        sigma = torch.exp(sigma) + 1e-15
        sigma = sigma * torch.sqrt(tau)  # Scale sigma with temperature
        
        # Apply temperature to mixture weights and ensure they sum to 1
        pi = F.softmax(pi/tau, dim=-1) + 1e-15
        
        return pi, mu, sigma
        

class MDNRNN(nn.Module):
    """
    Combined MDN-RNN model that predicts distributions over future latent states.
    
    Args:
        latent_dim: Dimension of the latent state vector
        action_dim: Dimension of the action vector
        hidden_dim: Dimension of the RNN hidden state
        num_gaussians: Number of Gaussian components in the mixture
        num_layers: Number of LSTM layers
    """
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians, num_layers=1):
        super(MDNRNN, self).__init__()
        
        self.rnn = RNN(latent_dim, action_dim, hidden_dim, num_layers)
        self.mdn = MDN(latent_dim, hidden_dim, num_gaussians)
        
    def forward(self, z, a, tau=1.0, h=None):
        """
        Forward pass through the combined MDN-RNN model.
        
        Args:
            z: Latent state tensor [batch_size, seq_len, latent_dim]
            a: Action tensor [batch_size, seq_len, action_dim]
            tau: Temperature parameter for controlling prediction sharpness
            h: Hidden state tuple (h_n, c_n) or None
            
        Returns:
            pi: Mixture weights [batch_size, seq_len, num_gaussians]
            mu: Gaussian means [batch_size, seq_len, num_gaussians, latent_dim]
            sigma: Gaussian std devs [batch_size, seq_len, num_gaussians, latent_dim]
            h: Updated hidden state tuple (h_n, c_n)
        """
        outs_rnn, h = self.rnn(z, a, h)
        pi, mu, sigma = self.mdn(outs_rnn, tau)
        
        return pi, mu, sigma, h


def sample_mdn(pi, mu, sigma):
    """
    Sample a latent vector from the mixture distribution.
    
    Args:
        pi: Mixture weights [1, 1, num_gaussians]
        mu: Gaussian means [1, 1, num_gaussians, latent_dim]
        sigma: Gaussian std devs [1, 1, num_gaussians, latent_dim]
        
    Returns:
        z: Sampled latent vector [1, 1, latent_dim]
    """
    # Select a Gaussian component based on mixture weights
    component_index = torch.multinomial(pi.view(-1), 1).item()  

    # Get the mean and std dev for the selected component
    selected_mu = mu[0, 0, component_index]   
    selected_sigma = sigma[0, 0, component_index]  

    # Sample from the Gaussian distribution
    epsilon = torch.randn(selected_mu.size(), device=mu.device)
    z = selected_mu + selected_sigma * epsilon

    return z.unsqueeze(0).unsqueeze(0)


def gaussian_nll_loss(pi, mu, sigma, z_next):
    """
    Compute the negative log-likelihood loss for a mixture of Gaussians.
    
    Args:
        pi: Mixture weights [batch, seq, num_gaussians]
        mu: Gaussian means [batch, seq, num_gaussians, latent_dim]
        sigma: Gaussian std devs [batch, seq, num_gaussians, latent_dim]
        z_next: Target latent vectors [batch, seq, latent_dim]
        
    Returns:
        nll: Scalar negative log-likelihood loss
    """
    z_next = z_next.unsqueeze(2)  # Add dim for num_gaussians: [batch, seq, 1, latent_dim]
    log_pi = torch.log(pi)
    
    # Create normal distributions for each mixture component
    normal_dist = Normal(loc=mu, scale=sigma)  # [batch, seq, num_gaussians, latent_dim]

    # Compute log probabilities for the target z_next
    log_prob = normal_dist.log_prob(z_next)  # [batch, seq, num_gaussians, latent_dim]

    # Sum log probabilities across the latent dimension
    log_prob = log_prob.sum(-1)  # [batch, seq, num_gaussians]

    # Use log-sum-exp trick for numerical stability
    log_sum_exp = torch.logsumexp(log_pi + log_prob, dim=-1)  # [batch, seq]

    # Mean negative log-likelihood
    nll = -log_sum_exp.mean()
    return nll

# %% [markdown]
# ## **3. Model Training and Evaluation**
# ---
# 
# ### Model Instantiation and Summary
# 

# %%
mdnrnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5)

print(mdnrnn)
total_params = sum(p.numel() for p in mdnrnn.parameters())
print(f"Number of parameters: {total_params:,}")

# %% [markdown]
# ### Training Function
# 
# This function trains the MDN-RNN model with gradient clipping for stability.
# 

# %%
import torch
from tqdm import tqdm 
import os 

def train_model(mdnrnn, dataloader, optimizer, num_epochs, device, save=True, tau=1.0, name='memory', path='checkpoints'):
    """
    Train the MDN-RNN model.
    
    Args:
        mdnrnn: The MDN-RNN model instance
        dataloader: DataLoader containing training data
        optimizer: Optimizer for parameter updates
        num_epochs: Number of training epochs
        device: Device to train on (cpu/cuda)
        save: Whether to save the model
        tau: Temperature parameter for MDN
        name: Name for the saved model file
    
    Returns:
        loss_history: List of average losses per epoch
    """
    mdnrnn.train()
    loss_history = []

    os.makedirs(path, exist_ok=True)

    # Apply gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), max_norm=1.0)
        
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        with tqdm(total=len(dataloader), desc=f"Epoch ({epoch+1}/{num_epochs})", unit="batch") as pbar:
            
            for z, z_next, actions, rewards, dones in dataloader:
                
                optimizer.zero_grad()
                z, z_next, a = z.to(device), z_next.to(device), actions.to(device)

                pi, mu, sigma, h = mdnrnn(z, a, tau=tau)
                
                # Calculate loss and track reconstruction quality                            
                loss = gaussian_nll_loss(pi, mu, sigma, z_next)
                recon_loss = torch.abs(z_next.unsqueeze(2) - mu).mean().item()

                # Backpropagation
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'NLL Loss': f'{loss.item():.4f} | Reconstruction Loss: {recon_loss:.4f}'})
    
        # Track epoch loss
        average_loss = total_loss / len(dataloader)
        loss_history.append(average_loss)
        
        pbar.set_postfix({'NLL Loss': f'{average_loss:.4f} | Reconstruction Loss: {recon_loss:.4f}'})
    
    if save:
        torch.save(mdnrnn.state_dict(), f'{path}/{name}.pth')
        print(f"Model saved to {path}/{name}.pth")
    
    return loss_history

# %% [markdown]
# ### Data Loading and Model Training
# 

# %%
from torch.utils.data import DataLoader
from utils import VAE
import torch 

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model dimensions
LATENT_DIM = 32
ACTION_DIM = 3

# Load the vision component (VAE)
vae_model = VAE(3, LATENT_DIM).to(device)
vae_model.load_state_dict(torch.load('models/vision_32.pth', map_location=device))

# Create dataset and dataloader
dataset = CarRacingDataset(h5_path='car_racing_data_10k.h5', transform=None, vae=vae_model, device=device)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

# Training hyperparameters
EPOCHS = 5
TAU = 1.0
NUM_GAUSSIAN = 5
HIDDEN_DIM = 256

# Initialize model and optimizer
model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
# Uncomment to load a pre-trained model
# model.load_state_dict(torch.load('models/memory_256.pth', map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
loss_history = train_model(model, dataloader, optimizer, num_epochs=EPOCHS, 
                          device=device, save=True, tau=TAU, name='memory_256')

# %%
# # Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
model.load_state_dict(torch.load('checkpoints/memory.pth', weights_only=False))


from models.vae import VAE
vae_model = VAE(3, 32).to(device)
vae_model.load_state_dict(torch.load('checkpoints/vae.pth', weights_only=False))

# %% [markdown]
# ## **4. Model Visualization**
# ---
# 
# ### Interactive Visualization
# 
# This function allows visual interaction with the model predictions in the Car Racing environment:
# 
# 

# %%
import gymnasium as gym
import pygame
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2

transform = transforms.Compose([
    transforms.Lambda(lambda img: img[:-12, :, :]),  # Crop the bottom 12 pixels
    transforms.ToPILImage(),                         # Convert to PIL image
    transforms.Resize((96, 96), transforms.InterpolationMode.LANCZOS),  # Resize
    transforms.ToTensor()                            # Convert to tensor and scale to [0,1]
])

def run_car_racing_rnn_mda(env_name, vae_model, mdnrnn, transform, device, scale=1, resolution=(150, 150), 
                           tau=1.0, save_video=False, video_filename="car_racing_rnn.avi"):
    """
    Run the Car Racing environment with MDN-RNN visualization.
    
    Args:
        env_name: Name of the Gym environment
        vae_model: Trained VAE model
        mdnrnn: Trained MDN-RNN model
        transform: Image transform function
        device: Device to run models on
        scale: Display scale factor
        resolution: Base resolution
        tau: Temperature parameter for sampling
        save_video: Whether to save a video of the run
        video_filename: Filename for the output video
    """
    # Initialize pygame for rendering
    pygame.init()
    resolution = (resolution[0] * 2 * scale, resolution[1] * scale)
    screen = pygame.display.set_mode(resolution)
    clock = pygame.time.Clock()  

    os.makedirs('videos', exist_ok=True)
    video_filename = os.path.join('videos', video_filename)
    
    action = np.zeros(3)  # Initialize action array
    
    def get_action(keys):
        """ Map keyboard input to actions """
        action[0] = -1.0 if keys[pygame.K_LEFT] else 1.0 if keys[pygame.K_RIGHT] else 0.0  # Steering
        action[1] = 1.0 if keys[pygame.K_UP] else 0.0  # Accelerate
        action[2] = 1.0 if keys[pygame.K_DOWN] else 0.0  # Brake
        return action
    
    # Initialize the environment
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (resolution[0], resolution[1]))

    running = True
    h = mdnrnn.rnn.init_hidden(1) 
    h = (h[0].to(device), h[1].to(device))
    
    cnt = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()  # Get current key states
        action = get_action(keys)        # Update action based on key presses

        # Environment step 
        obs, reward, done, info, _ = env.step(action)
        
        # Render and process the frame
        obs_tensor = transform(obs).unsqueeze(0).to(device)  # Transform frame to tensor

        with torch.no_grad():
            # Encode the frame using VAE
            mu, logvar = vae_model.encode(obs_tensor)
            z = vae_model.reparameterize(mu, logvar)  # Reparameterization trick
                
            # Generate predicted next image using RNN-MDN
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pi, mu, sigma, h = mdnrnn(z.unsqueeze(0), action_tensor, h=h, tau=tau)
            z_next = sample_mdn(pi, mu, sigma)
            
            # Decode MDN sampled latent vector
            reconstructed = vae_model.decode(z_next.squeeze(0))

        cnt+=1
        
        # Prepare images for display
        reconstructed = (reconstructed.squeeze(0).permute(2, 1, 0).cpu().numpy() * 255).astype(np.uint8)
        obs = (obs_tensor.squeeze(0).permute(2, 1, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Concatenate original and reconstructed images
        full_image = np.concatenate((obs, reconstructed), axis=0)
        full_image_resized = cv2.resize(full_image, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
        
        if save_video and video_writer is not None:
            video_writer.write(cv2.cvtColor(full_image_resized.transpose(1, 0, 2), cv2.COLOR_RGB2BGR))

        # Display the combined image
        clock.tick(30)
        pygame.surfarray.blit_array(screen, full_image_resized)
        pygame.display.flip()
        
        if done:
            obs, _ = env.reset()  # Reset environment if done
            h = mdnrnn.rnn.init_hidden(1) 
            h = (h[0].to(device), h[1].to(device))
    
    if save_video and video_writer is not None:
        video_writer.release()

    pygame.quit()
    env.close()


# Run the visualization
run_car_racing_rnn_mda(env_name="CarRacing-v3", vae_model=vae_model, mdnrnn=model, 
                      transform=transform, device=device, scale=4, tau=0.1, save_video=False)

# %% [markdown]
# ## **5. Model "Dreams": Exploring Autonomous Generation**
# ---
# 
# This visualization demonstrates the model's ability to generate its own predictions autonomously, beginning from a single real observation and then continuing with self-generated states:
# 

# %%
import gymnasium as gym
import pygame
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2

def run_car_racing_rnn_mda(env_name, vae_model, mdnrnn, transform, device, scale=1, resolution=(96, 96), tau=1.0, save_video=False, video_filename="car_racing_dream.avi"):
    # Initialize the environment
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    for _ in range(50):
        env.step(np.array([0,0,0]))
    # Initialize pygame for rendering
    pygame.init()
    resolution = (resolution[0] * scale, resolution[1] * scale)
    screen = pygame.display.set_mode(resolution)

    os.makedirs('videos', exist_ok=True)
    video_filename = os.path.join('videos', video_filename)
    
    action = np.zeros(3)  # Initialize action array
    
    def get_action(keys):
        """ Map keyboard input to actions """
        action[0] = -1.0 if keys[pygame.K_LEFT] else 1.0 if keys[pygame.K_RIGHT] else 0.0  # Steering
        action[1] = 1.0 if keys[pygame.K_UP] else 0.0  # Accelerate
        action[2] = 1.0 if keys[pygame.K_DOWN] else 0.0  # Brake
        return action
    
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (resolution[0], resolution[1]))

    running = True
    h = mdnrnn.rnn.init_hidden(1) 
    h = (h[0].to(device), h[1].to(device))
    
    cnt = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()  # Get current key states
        action = get_action(keys)        # Update action based on key presses
        
        ## Enviroment step 
        obs, reward, done, info, _ = env.step(action)
        
        obs_tensor = transform(obs).unsqueeze(0).to(device)  # Transform frame to tensor


        with torch.no_grad():
            
            # The fisrt latent vector comes from the enviroment
            # the next is the previous z generated from the LSTM-MDN
            if cnt ==0:
                mu, logvar = vae_model.encode(obs_tensor)
                z = vae_model.reparameterize(mu, logvar).unsqueeze(0) 
            else:
                z = z_next
                
            # Generate predicted next image  using RNN-MDA
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pi, mu, sigma, h = mdnrnn(z, action_tensor, h=h, tau=tau)
            z_next = sample_mdn(pi, mu, sigma)
            
            # Decode MDN sampled latent vector
            reconstructed = vae_model.decode(z_next.squeeze(0))

        cnt += 1
        
        # Prepare the reconstructed image for display
        reconstructed = (reconstructed.squeeze(0).permute(2, 1, 0).cpu().numpy() * 255).astype(np.uint8)
        reconstructed_resized = cv2.resize(reconstructed, (resolution[0], resolution[1]))

        if save_video and video_writer is not None:
            video_writer.write(cv2.cvtColor(reconstructed_resized.transpose(1, 0, 2), cv2.COLOR_RGB2BGR))

        # Display only the reconstructed image
        pygame.surfarray.blit_array(screen, reconstructed_resized)
        pygame.display.flip()
        
        if done:
            obs = env.reset()  # Reset environment if done
            h = mdnrnn.rnn.init_hidden(1) 
            h = (h[0].to(device), h[1].to(device))
    
    if save_video and video_writer is not None:
        video_writer.release()

    pygame.quit()
    env.close()


run_car_racing_rnn_mda(env_name="CarRacing-v3", vae_model=vae_model, mdnrnn=model, transform=transform, device=device, scale=6, tau=0.01, save_video=False)


