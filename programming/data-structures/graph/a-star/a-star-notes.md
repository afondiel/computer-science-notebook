# Pathfinding A* (star) - Notes

F  = G + H (calculated every time we create new node)

where : F : Total cost of the Node 
	G : Distance between current node and the start node 
	H : Heuristic (estimated distance from the current node to the end node)


Ex. From "Image A star.png"
	node (0) : starting position  
	node (19) : end position 
	node (4) : current position
	
	G= 4? 
	H= pythagore ? 58 (7^2 + 3^3 = 58^2 ===> 58) ?
	F= 62	? 
	
STEPS To build the algo : 
===

```
1. Add the starting square (or node) to the open list.
2. Repeat the following:
    A) Look for the lowest F cost square on the open list. We refer to this as the current square.
    B). Switch it to the closed list.
    C) For each of the 8 squares adjacent to this current square …
        If it is not walkable or if it is on the closed list, ignore it. Otherwise do the following.
        If it isn’t on the open list, add it to the open list. Make the current square the parent of this square. Record the F, G, and H costs of the square.
        If it is on the open list already, check to see if this path to that square is better, using G cost as the measure. A lower G cost means that this is a better path. If so, change the parent of the square to the current square, and recalculate the G and F scores of the square. If you are keeping your open list sorted by F score, you may need to resort the list to account for the change.
    D) Stop when you:
        Add the target square to the closed list, in which case the path has been found, or
        Fail to find the target square, and the open list is empty. In this case, there is no path.
3. Save the path. Working backwards from the target square, go from each square to its parent square until you reach the starting square. That is your path.
```
	
Pseudocode : 
===
```
// A* (star) Pathfinding
// Initialize both open and closed list
let the openList equal empty list of nodes
let the closedList equal empty list of nodes
// Add the start node
put the startNode on the openList (leave it's f at zero)
// Loop until you find the end
while the openList is not empty
    // Get the current node
    let the currentNode equal the node with the least f value
    remove the currentNode from the openList
    add the currentNode to the closedList
    // Found the goal
    if currentNode is the goal
        Congratz! You've found the end! Backtrack to get path
    // Generate children
    let the children of the currentNode equal the adjacent nodes
    
    for each child in the children
        // Child is on the closedList
        if child is in the closedList
            continue to beginning of for loop
        // Create the f, g, and h values
        child.g = currentNode.g + distance between child and current
        child.h = distance from child to end
        child.f = child.g + child.h
        // Child is already in openList
        if child.position is in the openList's nodes positions
            if the child.g is higher than the openList node's g
                continue to beginning of for loop
        // Add the child to the openList
        add the child to the openList
```
		
# References

- [Path & Motion Planning](https://en.wikipedia.org/wiki/Motion_planning)
- https://en.wikipedia.org/wiki/Pathfinding
- https://en.wikipedia.org/wiki/A*_search_algorithm
- https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
