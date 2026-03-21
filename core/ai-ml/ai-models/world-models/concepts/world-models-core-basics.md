# World Models Technical Notes
<!-- A rectangular image showing a simple world model concept: an agent (like a robot or AI character) observing a virtual environment through a "camera" eye, with internal thought bubbles representing a compressed mental model of the world (grid-like map with objects), arrows showing how observations update the model and how the model helps predict future actions and outcomes. -->

## Quick Reference
- **Definition**: A World Model is an internal representation that an AI agent builds of its environment, allowing it to predict what will happen next, plan actions, and make decisions without always needing to interact directly with the real world.
- **Key Use Cases**: Training AI for games (like playing Atari or racing games), robotics (simulating movements before trying them in reality), and autonomous systems that need to imagine future scenarios.
- **Prerequisites**: Basic understanding of neural networks and reinforcement learning concepts; no advanced math required.

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
A World Model is a learned neural network that compresses and simulates the environment an AI agent is operating in, acting like an internal "mental map" that predicts future states from current observations and actions.

### Why
World Models solve the problem of sample inefficiency in reinforcement learning by letting the agent "imagine" thousands of scenarios inside its model instead of repeatedly trying actions in the expensive real environment, leading to faster and safer learning.

### Where
World Models are used in game AI, robotics simulation, autonomous driving, and any domain where an agent needs to plan ahead or explore safely.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**: The agent observes the world → encodes observations into a compact latent representation → uses this representation to predict what happens if it takes certain actions → decodes predictions back into expected observations.
- **Key Components**:
  - **Encoder**: Compresses raw observations (images, sensor data) into a small latent vector.
  - **Dynamics Model**: Predicts how the latent state changes given an action.
  - **Decoder**: Reconstructs predicted observations from the latent state.
  - **Reward Predictor** (optional): Estimates future rewards directly from the latent state.
- **Common Misconceptions**:
  - World Models replace the real environment completely: They are approximations and work best when combined with some real interaction.
  - They are only for games: They are powerful in robotics and planning.
  - Building one requires massive data: Even simple versions can be trained on modest datasets.

### Visual Architecture
```mermaid
graph TD
    A[Raw Observation<br>(Image/Sensor Data)] -->|Encoder| B[Latent State z]
    B -->|Action a| C[Dynamics Model<br>Predict z_next]
    C -->|Decoder| D[Predicted Observation]
    B -->|Reward Predictor| E[Predicted Reward]
```
- **System Overview**: Observations are compressed into a latent state, which evolves with actions to predict future states and rewards.
- **Component Relationships**: The encoder creates the internal representation, the dynamics model simulates transitions, and the decoder lets the agent "imagine" outcomes.

## Implementation Details
### Basic Implementation
```python
import torch
import torch.nn as nn

class SimpleWorldModel(nn.Module):
    def __init__(self, obs_dim=784, latent_dim=32, action_dim=4):
        super().__init__()
        # Encoder: compress image to latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        # Dynamics: predict next latent given current latent + action
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        # Decoder: reconstruct observation from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
            nn.Sigmoid()
        )
    
    def forward(self, obs, action):
        z = self.encoder(obs)                    # Compress observation
        z_next = self.dynamics(torch.cat([z, action], dim=1))  # Predict next state
        obs_pred = self.decoder(z_next)          # Predict what it will look like
        return z, z_next, obs_pred

# Example usage
model = SimpleWorldModel()
dummy_obs = torch.randn(1, 784)      # e.g., flattened 28x28 image
dummy_action = torch.zeros(1, 4)     # one-hot or vector action
z, z_next, pred = model(dummy_obs, dummy_action)
print("Latent shape:", z.shape)
```
- **Step-by-Step Setup**:
  1. Install PyTorch: `pip install torch torchvision`.
  2. Define the three core networks (encoder, dynamics, decoder).
  3. Train by minimizing reconstruction error + prediction error.
- **Code Walkthrough**:
  - The encoder turns raw input into a compact state.
  - The dynamics model predicts how the state changes with an action.
  - The decoder reconstructs what the agent should see next.
- **Common Pitfalls**:
  - Latent dimension too small → poor reconstruction.
  - Training only on reconstruction without dynamics loss → model doesn't learn to predict actions.
  - Ignoring stochasticity → deterministic models struggle with uncertainty.

## Real-World Applications
### Industry Examples
- **Use Case**: Training a self-driving car in simulation before real roads.
- **Implementation Pattern**: Use camera images as observations, learn a world model to predict future frames and rewards (safety, progress).
- **Success Metrics**: Safe exploration in simulation, faster real-world transfer.

### Hands-On Project
- **Project Goals**: Build a simple world model for a 2D grid world or simple game.
- **Implementation Steps**:
  1. Use the code above as base.
  2. Train on a dataset of game screenshots + actions.
  3. Visualize predicted vs. actual next frames.
  4. Use the model inside a simple planner to choose actions.
- **Validation Methods**: Measure reconstruction quality (MSE) and prediction accuracy over multiple steps.

## Tools & Resources
### Essential Tools
- **Development Environment**: Python + PyTorch.
- **Key Frameworks**: Gymnasium (for environments), Stable Baselines3.
- **Testing Tools**: TensorBoard for visualizing latent spaces.

### Learning Resources
- **Documentation**: Original World Models paper (2018) by David Ha & Jürgen Schmidhuber.
- **Tutorials**: "World Models in PyTorch" on GitHub.
- **Community Resources**: r/MachineLearning, tinyML discussions.

## References
- "World Models" by Ha and Schmidhuber (2018).
- Dreamer series papers (DeepMind).
- "Planning with World Models" resources on arXiv.

## Appendix
### Glossary
- **Latent State (z)**: Compressed internal representation of the environment.
- **Dynamics Model**: Predicts how the world changes given an action.
- **Reconstruction Loss**: Measures how well the model can rebuild observations.

### Setup Guides
- Basic environment: `pip install torch gymnasium`.
- Test run: Load a simple Gym environment and feed observations to the model.

</xaiArtifact>