import torch
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
        self.hidden_dim = hidden_dim
        
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