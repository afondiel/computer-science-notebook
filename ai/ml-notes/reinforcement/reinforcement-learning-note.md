# Reinforcement learning (RL) - notes :robot:

## Intro 

A machine learning approach/paradigm which enables a model to learning patterns from data with/without labels based on reward (if the model correctly learns a pattern) or a penalty otherwise.
- A simple RL is modeled by Markov Decision Process (MDP)

RL provides a statistical framework based on two components :

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/800px-Reinforcement_learning_diagram.svg.png" width="300" style="border:2px solid #FFFFFF; padding:5px; margin:2px">

- Agent(a) : takes actions that changes an existing state(S) of the environment
- Environment (b) : transites to a new/future state based on changes taken by the agent and provides feedback to the agent
- feedback : can either be positive(+) or negative(-), after set of feedbacks or iterations the agent tries to learn and optimize the future actions in order to get a maxime of (+) feedbacks => reward (function)
- Quality/Q-function : a given action that produces a future reward based on current state

 Q-function : uses Bellman equation Q(s,a)
 
 ![Q-function](https://cdn-media-1.freecodecamp.org/images/s39aVodqNAKMTcwuMFlyPSy76kzAmU5idMzk)

Q-funtion/Q-Learning strategy : 
1. Value-based RL : Optimizes the maximum value function for a given action-state pair
2. Policy search-based RL : search for an optimal policy that's able to achiieve a maximum future reward
3. Actor-critic-based RL : hybrid strategy which combines value-based + policy-based to solve RL learning problems

for a high dimensinal environment a [Deep Reinforcement Learning](https://www.youtube.com/watch?v=zR11FLZ-O9M) is required
## Applications 
- Robotics 
- Self-driving cars 
- Strategy games : chess/Go/FrozenLake ...

## Tools / frameworks 
- OpenAI Gym
- TensorFlow
- Keras
- DeepMind Lab
- Pytorch
- Markov Chain/Markov Decision Process (MDP)


## References 

- [Reinforcement learning wiki](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
- [MDP](https://en.wikipedia.org/wiki/Markov_decision_process)
- [Stochastic Model](https://en.wikipedia.org/wiki/Stochastic_process)
- [Proba Theory](https://en.wikipedia.org/wiki/Probability_theory)
- [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation)
- [Sequential logic](https://en.wikipedia.org/wiki/Sequential_logic)
- [Finite-state machine](https://en.wikipedia.org/wiki/Finite-state_machine)
- [Dynamical_system](https://en.wikipedia.org/wiki/Dynamical_system)
- [implementing the a3c algorithm](https://medium.com/@shagunm1210/implementing-the-a3c-algorithm-to-train-an-agent-to-play-breakout-c0b5ce3b3405)



> *"It is difficult to free fools from the chains they revere." ~ Voltaire*

