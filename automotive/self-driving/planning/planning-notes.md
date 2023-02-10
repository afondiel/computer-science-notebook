# Planning - Notes

**Agenda**
- [What's planning ?](#whats-planning-)
- [Applications](#applications)
- [Tools and Frameworks](#tools-and-frameworks)
- [References](#references)


## What's planning ? 
![](https://media.licdn.com/dms/image/C5612AQGM2cTqtwJuIg/article-inline_image-shrink_1000_1488/0/1521484871946?e=1678924800&v=beta&t=pkrKCMAM76QDzEgzLKn-acdrjFGivPKiW158OEJUqgQ)

Planning Layers : 
- mission planning : which street to take to achieve a mission goal. 
- behavioral planning : when to change lanes and precedence at intersections and performs error recovery maneuvers.
- motion planning : selects actions to avoid obstacles while making progress
toward local goals


Planning multiple dimensions : 
- Location
- Orientation
- Direction of travel
  
## Applications
- Self-driving vehicles
- Robotics

## Tools and Frameworks
Sensors : 
- GPS
- IMU
- Lidar
... 

### Motion Planning algorithms

- A* (star) search 
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
- Dijkstra

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


- Kruskal's algorithm ? 
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
  
- Others Algorithms 
  - Breadth-first search(BFS)
  - Suboptimal A* search.
  - 
# References

- [Machine planning](https://en.wikipedia.org/wiki/Machine_planning)
- [Boss Autonomous Driving](https://github.com/afondiel/Self-Driving-Cars-Specialization-Coursera/blob/main/Course1-Introduction-to-Self-Driving-Cars/resources/Boss-autonomous-driving-pres-DARPA-Urban-Challenge-2007-by-journal-of-robotics-2008.pdf)

 - [Perception Planning and Control for Self-Driving System Based on On-board Sensors](https://www.researchgate.net/publication/344734310_Perception_Planning_and_Control_for_Self-Driving_System_Based_on_On-board_Sensors/link/609c9edd299bf1259ece7fe0/download)

