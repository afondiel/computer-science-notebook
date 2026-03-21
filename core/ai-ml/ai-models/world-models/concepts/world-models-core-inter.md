# World Models Technical Notes
<!-- A rectangular image showing an intermediate world model architecture: an agent observing pixel-based environments, encoding into a latent space with variational autoencoders, a recurrent dynamics model predicting future latent states, a reward predictor, and a planner using imagined rollouts for action selection, with visualizations of predicted vs real trajectories. -->

## Quick Reference
- **Definition**: World Models are learned internal simulations of the environment that allow an agent to predict future states, plan actions, and improve sample efficiency in reinforcement learning.
- **Key Use Cases**: Model-based reinforcement learning for games, robotics simulation-to-real transfer, and safe exploration in complex environments.
- **Prerequisites**: Solid understanding of neural networks, autoencoders, RNNs/LSTMs, and basic reinforcement learning concepts.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
### What
World Models learn a compressed, predictive representation of the environment, enabling the agent to "imagine" future trajectories inside the model rather than only in the real world.

### Why
They dramatically improve sample efficiency and safety by allowing thousands of imaginary rollouts during planning, reducing the need for dangerous or expensive real-world interactions.

### Where
World Models are widely used in game AI (Atari, MuJoCo), robotics (sim-to-real), autonomous driving, and any RL setting where data collection is costly.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**: Combine a variational autoencoder (VAE) for perception with a recurrent dynamics model to predict future latent states, enabling model-based planning via imagined trajectories.
- **Key Components**:
  - **VAE Encoder/Decoder**: Compresses high-dimensional observations into a stochastic latent space.
  - **Recurrent Dynamics Model (MDN-RNN)**: Predicts the distribution of the next latent state given current latent and action.
  - **Reward & Termination Predictors**: Estimate future rewards and episode ends from latent states.
  - **Controller/Planner**: Uses the world model for rollouts to select optimal actions (CEM, MPC, or policy networks).
- **Common Misconceptions**:
  - World Models are purely deterministic: Most use stochastic latents (VAE) to handle uncertainty.
  - Planning is always explicit: Many modern versions train an actor-critic directly inside the imagined world.
  - They replace all real data: Hybrid approaches (real + imagined data) usually perform best.

### Visual Architecture
```mermaid
graph TD
    A[Raw Observation o_t] -->|VAE Encoder| B[Latent z_t ~ N(μ,σ)]
    B -->|Action a_t| C[MDN-RNN Dynamics<br>→ p(z_{t+1}|z_t,a_t)]
    C -->|Sample z_{t+1}| D[Reward Predictor r_t]
    D -->| imagined trajectory| E[Planner / Controller]
    B -->|VAE Decoder| F[Reconstructed Observation]
```
- **System Overview**: Observations are encoded into stochastic latents, the dynamics model predicts future latents, and the planner uses imagined rollouts for decision making.
- **Component Relationships**: The VAE provides perception, the RNN provides temporal dynamics, and the planner leverages both for action selection.

## Implementation Details
### Intermediate Patterns
```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class WorldModel(nn.Module):
    def __init__(self, obs_dim, latent_dim=32, action_dim=4, hidden_dim=256):
        super().__init__()
        # VAE components
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # μ and logσ
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid()
        )
        
        # Recurrent Dynamics (simple RNN for illustration)
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.dynamics_head = nn.Linear(hidden_dim, latent_dim * 2)  # next μ, logσ
        
        # Reward predictor
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def encode(self, obs):
        h = self.encoder(obs)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # reparameterization
        return z, mu, logvar
    
    def predict_next(self, z, action):
        rnn_input = torch.cat([z, action], dim=-1).unsqueeze(1)
        h, _ = self.rnn(rnn_input)
        next_params = self.dynamics_head(h.squeeze(1))
        next_mu, next_logvar = next_params.chunk(2, dim=-1)
        return next_mu, next_logvar
    
    def forward(self, obs, action):
        z, mu, logvar = self.encode(obs)
        next_mu, next_logvar = self.predict_next(z, action)
        next_z = next_mu + torch.exp(0.5 * next_logvar) * torch.randn_like(next_mu)
        reward = self.reward_head(next_mu)  # simplified
        recon = self.decoder(z)
        return recon, next_z, reward, mu, logvar

# Training loop sketch
model = WorldModel(obs_dim=784, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    obs, action, next_obs, reward = batch
    recon, next_z, pred_reward, mu, logvar = model(obs, action)
    
    # Losses: reconstruction + KL + dynamics + reward
    recon_loss = nn.MSELoss()(recon, next_obs)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    reward_loss = nn.MSELoss()(pred_reward, reward)
    
    loss = recon_loss + 0.1 * kl_loss + reward_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
- **Design Patterns**:
  - **VAE + RNN Hybrid**: Perception via VAE, temporal modeling via recurrent dynamics.
  - **Imagined Rollouts**: Sample multiple futures inside the model for planning.
  - **Latent Space Control**: Train a controller directly in the learned latent space.
- **Best Practices**:
  - Use β-VAE to balance reconstruction and disentanglement.
  - Add stochasticity in dynamics for realistic uncertainty modeling.
  - Regularly mix real and imagined data during training.
- **Performance Considerations**:
  - Latent dimension trade-off: too small loses information, too large slows planning.
  - Use GPU acceleration for batch rollouts.
  - Monitor prediction horizon — models degrade over long imaginations.

## Real-World Applications
### Industry Examples
- **Use Case**: Sim-to-real transfer for robotic manipulation.
- **Implementation Pattern**: Train world model in simulation, fine-tune dynamics on real data, plan in latent space.
- **Success Metrics**: High success rate with few real-world samples.

### Hands-On Project
- **Project Goals**: Build a world model for a simple Gym environment (e.g., CartPole or CarRacing).
- **Implementation Steps**:
  1. Collect rollouts using a random or pretrained policy.
  2. Implement VAE + RNN world model as shown.
  3. Train with reconstruction + KL + reward losses.
  4. Use CEM or a small policy network inside imagined rollouts for planning.
- **Validation Methods**: Measure prediction accuracy over 10–20 steps and final task performance.

## Tools & Resources
### Essential Tools
- **Development Environment**: PyTorch, Gymnasium.
- **Key Frameworks**: Stable-Baselines3, DreamerV2/V3 implementations.
- **Testing Tools**: TensorBoard for latent visualization.

### Learning Resources
- **Original Paper**: "World Models" by Ha & Schmidhuber (2018).
- **Dreamer Series**: DeepMind's DreamerV2/V3 papers.
- **Tutorials**: "Building World Models in PyTorch" on GitHub.

## References
- Ha, D., & Schmidhuber, J. (2018). World Models.
- Hafner et al., Dreamer: Mastering Atari with World Models (2020+ series).
- "Planning to Explore via Self-Supervised World Models".

## Appendix
### Glossary
- **Latent Dynamics**: Prediction of future states in compressed space.
- **Imagined Rollouts**: Virtual trajectories generated inside the world model.
- **β-VAE**: Variational autoencoder with disentanglement parameter.

### Setup Guides
- Install dependencies: `pip install torch gymnasium stable-baselines3`.
- Start with simple Gym environments before moving to pixel-based ones.

</xaiArtifact>