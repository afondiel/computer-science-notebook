# Graph Neural Network - Notes

## Table of Contents (ToC)
- [Introduction](#introduction)
  - [What's Graph Neural Network?](#whats-graph-neural-network)
  - [Key Concepts and Terminology](#key-concepts-and-terminology)
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [Graph Neural Network Architecture Pipeline](#graph-neural-network-architecture-pipeline)
  - [How Graph Neural Networks Work?](#how-graph-neural-networks-work)
  - [Types of Graph Neural Networks](#types-of-graph-neural-networks)
  - [Some Hands-on Examples](#some-hands-on-examples)
- [Tools & Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)

## Introduction
Graph Neural Networks (GNNs) are deep learning models that operate on graph structures to capture dependencies and relationships between data points.

### What's Graph Neural Network?
- A type of neural network designed for processing graph-structured data.
- Works on nodes, edges, and their connections.
- Enables learning on non-Euclidean data (i.e., networks, social graphs).

### Key Concepts and Terminology
- **Node**: Basic entity in a graph representing an object or point.
- **Edge**: Connection between nodes, representing relationships.
- **Graph Convolution**: Operation to aggregate node information.
- **Message Passing**: The process of information exchange between nodes.

### Applications
- Social network analysis (e.g., friend recommendations).
- Biological networks (e.g., protein structure prediction).
- Traffic and infrastructure networks (e.g., route optimization).
- Recommender systems based on relational data.

## Fundamentals

### Graph Neural Network Architecture Pipeline
- Input: Graph data (nodes, edges).
- Graph convolution layers to aggregate node features.
- Message passing layers for relational learning.
- Output layer for classification or regression tasks.

### How Graph Neural Networks Work?
- Nodes exchange information with their neighbors (message passing).
- Feature aggregation is done iteratively across the graph.
- Final layer outputs task-specific predictions (e.g., node classification).

### Types of Graph Neural Networks
- **Graph Convolutional Networks (GCN)**: Uses convolution to aggregate node features.
- **Graph Attention Networks (GAT)**: Incorporates attention mechanisms in message passing.
- **Graph Recurrent Networks (GRN)**: Uses recurrent neural networks for graph data.
  
### Some Hands-on Examples
- Node classification on citation networks.
- Link prediction between users in a social graph.
- Graph classification for molecule structure data.

## Tools & Frameworks
- **PyTorch Geometric**: A framework for GNNs in PyTorch.
- **DGL (Deep Graph Library)**: High-performance GNN library.
- **Spektral**: Keras-based framework for GNNs.

## Hello World!
```python
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load dataset (e.g., Cora)
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN()
```

## Lab: Zero to Hero Projects
- **Project 1**: Implement node classification on a citation network.
- **Project 2**: Build a recommender system using GNNs for a social media platform.
- **Project 3**: Analyze molecular graphs for drug discovery.

## References
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks.
- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs.
- Zhou, J., et al. (2020). Graph neural networks: A review of methods and applications.
