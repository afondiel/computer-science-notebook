# Markov Decision Process(MDP) - notes
 
![MDP model](mdp.png)

[MDP Framework](https://en.wikipedia.org/wiki/Markov_decision_process)

A MDP is a mathematical models used to make decisions in situations where outcomes are partly random and partly under the control of a decision maker.

## The MDP Model

- `Agent` :   observes a state, St, from the environment at time t and interacts with the environment, by taking an action A(t). This action causes the Environment to transition to a new state, S(t+1),
- `Environment` : transites to a new/future state based on changes taken by the agent and provides feedback to the agent
- `State` : sets of state (S(t) : present state, R(t+1) : next state)
- `Action` : sets of actions perform by the agent

- `Reward` : success/penality (R(t) : present reward, R(t+1) : next reward)
- `Transition funtion - T[s,a,s']` : environment condition to the next state
- ` Policy function - Pi(s)` : function that computes the reward 
- `Optimal Policy function - Pi*(s)` : policy that maximizes the reward 


## The MDP Proprieties

- “Markov” generally means that given the present state, the future and the past are independent
- For Markov decision processes, “Markov” means action outcomes depend only on the current state
- MDPs are non-deterministic search problems
  - One way to solve them is with expectimax search
  - We’ll have a new tool soon
 
## Application
- Artificial Intelligence
- Reinforcement Learning 
- Robotics


## Tools/Framework

- Probabilities / statistics

# References 

- [UC Berkeley CS188 - MDP I](http://ai.berkeley.edu/slides/Lecture%208%20--%20MDPs%20I/SP14%20CS188%20Lecture%208%20--%20MDPs%20I.pptx)
- [UC Berkeley CS188 - MDP II](http://ai.berkeley.edu/slides/Lecture%209%20--%20MDPs%20II/SP14%20CS188%20Lecture%209%20--%20MDPs%20II.pptx)

- [Reinforcement Learning 3: Markov Decision Processes and Dynamic Programming - DeepMind](https://www.youtube.com/watch?v=hMbxmRyDw5M)

- [Markov Decision Processes 1 - Value Iteration | Stanford CS221: AI (Autumn 2019)](https://www.youtube.com/watch?v=9g32v7bK3Co&t=739s)

- [Markov Decision Processes 2 - Reinforcement Learning | Stanford CS221: AI (Autumn 2019)](https://www.youtube.com/watch?v=HpaHTfY52RQ)

- [GeeksforGeeks](https://www.geeksforgeeks.org/markov-decision-process/)