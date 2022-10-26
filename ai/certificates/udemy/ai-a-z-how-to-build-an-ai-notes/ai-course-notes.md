# Artificial Intelligence A-Z: Learn How To Build An AI - Notes


## Section 1: Welcome to the course  

### 1. Why AI ? 
- It's the best time to be in AI ! 
- Applications : Self-driving cars, medicine, heavy machanery, customer service ...
- Moore's Law : computer power will double every 2yrs 
    => Strategy games :
        => Chess : DeepMind vs Gary Kasparov (1997)
        => Go : AlphaGo vs Lee Sedol (2016)
- Game to train AI => confined environment then apply that to solve the real world problems/business

### 2. Course structure : OK
- course resources :  https://www.tfcertification.com/pages/artificial-intelligence
    2.1 BONUS : Learning Paths
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

