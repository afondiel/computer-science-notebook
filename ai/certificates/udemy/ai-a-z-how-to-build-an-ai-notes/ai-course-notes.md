# Artificial Intelligence A-Z: Learn How To Build An AI - Notes


## Section 1: Welcome to the course  

### 1. Why AI ? 
- It's the best time to be in AI ! 
- Applications : Self-driving cars, medicine, heavy machanery, customer service ...
- Moore's Law : computer power will double every 2yrs 
    - Strategy games :
        - Chess : DeepMind vs Gary Kasparov (1997)
        - Go : AlphaGo vs Lee Sedol (2016)
- Game to train AI : confined environment then apply that to solve the real world problems/business

### 2. Course structure : OK
- course resources & code template :  https://www.tfcertification.com/pages/artificial-intelligence
- 2.1 BONUS - Learning Paths :  https://sdsclub.com/learning-paths/ai-engineer/
### 4. Anaconda installation : OK 
### 5. Extra Materials : sdsclub.com/artificial-intelligence-a-z-learn-how-to-build-an-ai
    5.1 - QA : https://datascience.stackexchange.com/
    5.2 - To Becoming A Better Data Scientist  : interaction & feedbacks, help others, debugging, purpose ...
    5.3 - Discord for feedback 

## Section 2: - Part 0 - Fundamentals Of Reinforcement Learning

### 6. the problem setup :

How humans learn ? 
- The thing is we already know how we, humans, learn. 
- We understand the concepts of intrinsic and extrinsic rewards that guide us to become better at things. 
- For example, if you're playing bowling - you know that you need to hit the pins and a strike is a perfect shot.
- We know this because we are rewarded with points for hitting the pins down and we are "punished" with no points when your ball ends in the ditch. 
- So your brain projects those conditions of the environment onto your actions and that's how you know when you are doing good and when - not so much. 
- And that's how we learn.

How AI learns ? 
- But how do we explain that to an Al? 
    - Reinforcement Learning : learning processes similar Conceptually to human  

## Section 3: Q-Learning intuition

### 7. plan of attack

What we will learn in this section: 
- What is Reinforcement Learning? 
- The Bellman Equation 
- The "Plan" 
- Markov Decision Process (MDP) 
- Policy vs Plan 
- Adding a "Living Penalty" 
- Q-Learning Intuition 
- Temporal Difference 

### 8. What is reinforcement learning ?

Model : 
- Environment : the state of environment will change, it will also get a reward based on the actions taken by the agent
- Agent : performs action in the environment
- the number of iterations enable the Angent to learn about its environment and optimize which actions lead to bad rewards and unfavorable states 
- actions should be taken at the correct points in time (sequentual ? one after anoter ?)   
- reward : +1 for good action  ou -1 bad action/command 

- Ex of ai learning process : training a robotdog how to walk

Pre-programmed learning vs reinforcement learning 
- pre-programmed learning :  actions are programed in advance (move forward, left, right ...)
- reinforcement learning : there is no an hard-coded algorithm into the dog, there is a reinforcement model 
    - move towards a specific target, it gets a reward (+1/-1)
    - throught repetition it undertands how it can walk 
    - by numerous iteration it can optimize things on it own and  we get a better results 

- Addictional reading : 
    - [research paper by Richard Sutton et al. (1998)](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.7692)

### 9. The Bellman Equation

Invented by Richard Ernest Bellman, a Mathematician who came up w/ the Dynamic programing concept (today called RL) in 1953

Concepts : 
- s - State (where the agent is )
- a - Action (taken by the agent)
- R - Reward (got by the agent for entering into a certain state)
- Y(gamma) - Discount factor

Actions/List of actions are often associated with states.

How it works ? 

|      |       |       |Target | 
|------|-------|-------|-------|
|      |       |       |  Fire |
|      |   X   |       |       |
|   A  |       |       |       |

if A get reach the target => R = +1
If A == X => R=-1 
If A == Fire => R=-1 

each time the Agend(A) takes an action the state also change. 
The challenge is to make the Agent take the actions that lead to a positive reward based on previous states results(valuables state). 

How to calculate Valuable States (V)? 
- the idea is to go a backward from the target where a R=+1 and create pathway to the starting point. Then, detect every adjacent square or state that lead to target and value it V=1, and repeat the process until the full path is recreated from the starting point.
- Build the equation that helps the agent go through the maze:
    - look at reward, then the preceding state give a value of equal reward and so on to create a pathway.
    - the problem with approach is that if the Agent starts from any other state than the starting point it doesnot know anymore how to reach the target(Reward)

**Solution : dynamique programming the Bellman Equation**

    V(s) = max[R(s, a) + Y*V(s')]

where : s' - the next/future state
- Y : solve the situation where the agent doesn't know which way to go.
- The value of Y is chosen under certain conditions(ex: 0.9)
- Also we calculate the maximum value in each state until the starting point. 
- The pattern is that the preceding state < than the actual state. The closer you are to the finish line the more valuable that state is.
- With this concept it's easier to know which direction the agent must go.

### 10. The "Plan"

- create a navigation map of AI.
- we can replace the state value with the arrow indicating the direction to the finish line
- plan different from "policy"

### 11. Markov Decision Process (MDP)

The navigation map is not that simple that's where MDP comes to play.

- Deterministic search : always produce the same output from a given starting condition or initial state (ex : if the agent is told to go up there's a probability of 100% it goes in this direction)
- Non-deterministic search(stochastic process) : multiple possible outcomes from a given starting condition or initial state (ex : if the agent is told to go up there's a probability of 80% it goes up, 10% left or 10% right)  
    - How randomness affects an environment and how to deal w/ it 
    - Using Markov Decision Process

What's Markov Process ?
>A stochastic process has the Markov property if the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state. not on the sequence of events that preceded it. A process with this property is called a **Markov process** (src : wikipedia)

What's Markov Decision Processes?
> Markov Decision Processes (MDPs) provide a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.(src: wikipedia)

- A MDP is a version of Bellman equation but sophisticated

        V(s) = max[R(s, a) + Y*Sum(P(s,a, s')*V(s'))]

where : 
- P(s,a,s') : is the Probability of the action to go the actual state to a future state
- Sum : sum of the probability of different actions available for every new state possibility
- s' : the new/future state

Addictional resources :
- A survey of Applications of Markov Decision Processes by D. J. White(1993)
- Link : http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications3.pdf








## Section 4: Q-Learning Visualization
## Section 5: - Part 1 - Deep Q-Learning
## Section 6: Deep Q-Learning Intuition
## Section 7: Deep Q-Learning Implementation
## Section 8: Deep Q-Learning Visualization
## Section 9:  Part 2 - Deep Convolutional Q-Learning
## Section 10: Deep Convolutional Q-Learning Intuition
## Section 11: Deep Convolutional Q-Learning Implementation
## Section 12: Deep Convolutional Q-Learning Visualization
## Section 13: - Part 3 - A3C
## Section 14: A3C Intuition
## Section 15: A3C Implementation
## Section 16: ASC Visualization
## Section 17: Annex 1: Artificial Neural Networks
## Section 18: Annex 2: Convolutional Neural Networks
## Section 19: Bonus Lectures

# References : 

- Additional Resource
https://www.tfcertification.com/pages/artificial-intelligence

- AI wikipedia : 
https://en.wikipedia.org/wiki/Artificial_intelligence

- DeepMind : 
https://www.deepmind.com/tags/reinforcement-learning?fc63e648_page=8

- Reinforcement learning Implementation with TensorFlow : 
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

- IBM - What Is an AI Engineer? (And How to Become One)
https://www.coursera.org/articles/ai-engineer