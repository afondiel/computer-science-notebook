# Motion Planning - Notes

**Agenda**
- [What's Planning?](#whats-planning)
- [Planning Architecture Pipeline](#planning-architecture-pipeline)
- [Applications](#applications)
- [Tools and Frameworks](#tools-and-frameworks)
- [Motion Planning algorithms](#motion-planning-algorithms)
- [Mission Planner](#mission-planner)
- [Behaviour Planner](#behaviour-planner)
- [Local Planner](#local-planner)
- [Motion Planning Datasets \& Libraries](#motion-planning-datasets--libraries)
- [References](#references)


## What's Planning? 

![planning](https://1.bp.blogspot.com/-SkP0AfhlI9w/YEgQjwDgaSI/AAAAAAAAE_c/xCPqIwl23Yk7vcSijM4mrkS-55WmJ7IFwCNcBGAsYHQ/s16000/busy_street_3.png)

Src: @[Waymo](https://waymo.com/blog/2021/03/expanding-waymo-open-dataset-with-interactive-scenario-data-and-new-challenges.html)

The motion planning problem is the task of navigating the ego vehicle to its destination in a safe and comfortable manner while following the rules of the road.


## Planning Architecture Pipeline

![](https://github.com/afondiel/Self-Driving-Cars-Specialization/blob/main/Course1-Introduction-to-Self-Driving-Cars/resources/w2/img/l3-sw-archi-motion-planning2.png?raw=true)

- `Mission planning` : which street to take to achieve a mission goal. 
- `Behavioral planning` : when to change lanes and precedence at intersections and performs error recovery maneuvers.
- `Motion planning` : selects actions to avoid obstacles while making progress toward local goals

### Planning multiple dimensions
- Location
- Orientation
- Direction of travel (DoT)

## Applications

- Self-driving vehicles
- Robotics
- Drones

## Tools and Frameworks

- networkx
- osmnx
- [CVPR 2021: Forecasting for Motion Planning](https://www.youtube.com/watch?v=bnmrXt1g3aQ)
... 

## State-of-the-art Motion Planning Approaches

```
- Global Planner
  |
  |- Long-Term Planner
  |
  |- Short-Term Planner
      |
      |- Local Planner
          |
          |- Control Stack
```

- **Global planner (Long-term planner)** : map + static obstacles 
  - Rule Based Planning (pipeline method)
  - Predictive planning 
    - Imitation learning
    - Reinforcement learning 
    - Parallel learning
  - Trajectory planning
  - Graph Based Planning
  - Probabilistic Graph Based Planning
  - Optimization Based Planning
    - Linear Programming
    - NonLinear Programming 

- **Local planner (Short-term planner)**: dynamic obstacles
  - Reactive planning (trajectory roll-out planner)


## Motion Planning Algorithms

## Categories

- Grid-based search
- Interval-based search
- Geometric algorithms
- Artificial potential fields
- Sampling-based algorithms
- ...

**A\* (star) Search**

```
OPEN<-{1}
past_cost[1]<-0, past_cost[node]<-infinity for node €{2,...,N} 
while OPEN is not empty do
    current   first node in OPEN, remove from OPEN
    add current to CLOSED
    if current is in the goal set then
        return SUCCESS and the path to current
    end if
    for each nbr of current not in CLOSED do
        tentative_past_cost   past_cost[current]+cost[current,nbr]
        if tentative past cost < past cost[nbr] then
            past_cost[nbr]   tentative_past_cost
            parent[nbr]   current
            put (or move) nbr in sorted list OPEN according to
                est_total_cost[nbr]   past_cost[nbr] + heuristic_cost_to_go(nbr)
        end if
    end for
end while
return FAILURE
```

**Dijkstra**

```
 1  function Dijkstra(Graph, source):
 2      
 3      for each vertex v in Graph.Vertices:
 4          dist[v] ← INFINITY
 5          prev[v] ← UNDEFINED
 6          add v to Q
 7      dist[source] ← 0
 8      
 9      while Q is not empty:
10          u ← vertex in Q with min dist[u]
11          remove u from Q
12          
13          for each neighbor v of u still in Q:
14              alt ← dist[u] + Graph.Edges(u, v)
15              if alt < dist[v]:
16                  dist[v] ← alt
17                  prev[v] ← u
18
19      return dist[], prev[]
```
  - [C-Implementation](https://github.com/afondiel/research-notes/tree/master/programming/data%20structures/graph)

**Kruskal's algorithm ?**

```
algorithm Kruskal(G) is
    F:= ∅
    for each v ∈ G.V do
        MAKE-SET(v)
    for each (u, v) in G.E ordered by weight(u, v), increasing do
        if FIND-SET(u) ≠ FIND-SET(v) then
            F:= F ∪ {(u, v)} ∪ {(v, u)}
            UNION(FIND-SET(u), FIND-SET(v))
    return F
```
- [C implementation](https://github.com/afondiel/research-notes/tree/master/programming/data%20structures/graph)
  
**Breadth First Search (BFS)**

![bfs](https://github.com/afondiel/Self-Driving-Cars-Specialization-Coursera/blob/main/Course4-Motion-Planning-for-Self-Driving-Cars/resources/w3/img/l1-bfs0.png?raw=true)

- **D* (D-star)**: @TODO
- **Depth-First Search (DFS)** uses a last-in-first-out (LIFO) stack instead of a queue for the open set. 
- **Suboptimal A* search**
- **Rapidly-exploring random tree**
- **Probabilistic roadmap**


## Motion Planning Datasets & Libraries
- [Waymo open dataset](https://waymo.com/open/about)
- [nuPlan - Motional](https://www.nuscenes.org/nuplan)
- [The Open Motion Planning Library](https://ompl.kavrakilab.org/)

## Hello World!
@TODO

## References

Wikipedia:

- [Motion planning](https://en.wikipedia.org/wiki/Motion_planning)
- [Navigation](https://en.wikipedia.org/wiki/Navigation)

Courses:
- [Motion Planning Course- Self-Driving Cars Specialization of University of Toronto](https://github.com/afondiel/Self-Driving-Cars-Specialization-Coursera/tree/main/Course4-Motion-Planning-for-Self-Driving-Cars)

DARPA Challenge:
- [Boss Autonomous Driving](https://github.com/afondiel/Self-Driving-Cars-Specialization-Coursera/blob/main/Course1-Introduction-to-Self-Driving-Cars/resources/Boss-autonomous-driving-pres-DARPA-Urban-Challenge-2007-by-journal-of-robotics-2008.pdf)

MathWorks - MATLAB
- [Motion Planning with MATLAB](https://www.mathworks.com/campaigns/offers/motion-planning-with-matlab.html)

- [Choose Path Planning Algorithms for Navigation](https://www.mathworks.com/help/nav/ug/choose-path-planning-algorithms-for-navigation.html)
- [Motion Planning](https://www.mathworks.com/help/nav/motion-planning.html)

Academia:

- [6 Trajectory Generation - Clemson University](https://opentextbooks.clemson.edu/wangrobotics/chapter/trajectory-generation/)

- [Perception Planning and Control for Self-Driving System Based on On-board Sensors](https://www.researchgate.net/publication/344734310_Perception_Planning_and_Control_for_Self-Driving_System_Based_on_On-board_Sensors/link/609c9edd299bf1259ece7fe0/download)