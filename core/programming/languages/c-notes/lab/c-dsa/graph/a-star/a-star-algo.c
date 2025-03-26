/**
 * @file astar.c
 * @brief A* Algorithm implementation for pathfinding in a maze.
 */

#include <stdio.h>
#include <stdlib.h>

#define MAX_ROWS 10
#define MAX_COLS 10

/**
 * @struct Position
 * @brief Represents a position in the maze.
 */
typedef struct {
    int row; /**< The row index of the position. */
    int col; /**< The column index of the position. */
} Position;

/**
 * @struct Node
 * @brief Represents a node in the A* search algorithm.
 */
typedef struct Node {
    struct Node* parent; /**< Pointer to the parent node. */
    Position position;  /**< The position of the node. */
    int g;              /**< The cost from the start node to this node. */
    int h;              /**< The heuristic value from this node to the goal node. */
    int f;              /**< The total cost of the node (g + h). */
} Node;

/**
 * @brief Creates a new node with the given parent and position.
 * @param parent The parent node.
 * @param position The position of the node.
 * @return The newly created node.
 */
Node* createNode(Node* parent, Position position);

/**
 * @brief Calculates the Manhattan distance between two positions.
 * @param position1 The first position.
 * @param position2 The second position.
 * @return The Manhattan distance between the two positions.
 */
int getHeuristic(Position position1, Position position2);

/**
 * @brief Checks if two positions are equal.
 * @param position1 The first position.
 * @param position2 The second position.
 * @return 1 if the positions are equal, 0 otherwise.
 */
int isEqual(Position position1, Position position2);

/**
 * @brief Gets the adjacent positions of a given position.
 * @param position The position to get the adjacent positions of.
 * @return An array of adjacent positions.
 */
Position* getAdjacentPositions(Position position);

/**
 * @brief Filters out positions that are not walkable in the maze.
 * @param maze The maze.
 * @param positions The array of positions to filter.
 * @param count The number of positions in the array.
 * @param row The number of rows in the maze.
 * @param col The number of columns in the maze.
 * @return An array of filtered positions.
 */
Position* filterWalkablePositions(int maze[MAX_ROWS][MAX_COLS], Position* positions, int count, int row, int col);

/**
 * @brief Appends two arrays of positions.
 * @param positions1 The first array of positions.
 * @param count1 The number of positions in the first array.
 * @param positions2 The second array of positions.
 * @param count2 The number of positions in the second array.
 * @return The appended array of positions.
 */
Position* appendPositions(Position* positions1, int count1, Position* positions2, int count2);

/**
 * @brief Checks if a position is in the given list of positions.
 * @param position The position to check.
 * @param positions The array of positions to search in.
 * @param count The number of positions in the array.
 * @return 1 if the position is in the list, 0 otherwise.
 */
int isInList(Position position, Position* positions, int count);

/**
 * @brief Removes a position from the given list of positions.
 * @param position The position to remove.
 * @param positions The array of positions.
 * @param count The number of positions in the array.
 */
void removeFromList(Position position, Position* positions, int* count);

/**
 * @brief Finds the path from the start position to the end position in the given maze.
 * @param maze The maze.
 * @param row The number of rows in the maze.
 * @param col The number of columns in the maze.
 * @param start The start position.
 * @param end The end position.
 * @return An array of positions representing the path from start to end, or NULL if no path is found.
 */
Position* astar(int maze[MAX_ROWS][MAX_COLS], int row, int col, Position start, Position end);

/**
 * @brief Prints the path.
 * @param path The array of positions representing the path.
 * @param count The number of positions in the array.
 */
void printPath(Position* path, int count);

/**
 * @brief Main entry point of the program.
 */
int main() {
    int maze[MAX_ROWS][MAX_COLS] = {
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    int row = 10;
    int col = 10;

    Position start = {0, 0};
    Position end = {7, 6};

    Position* path = astar(maze, row, col, start, end);
    if (path != NULL) {
        printPath(path, row * col);
        free(path);
    } else {
        printf("No path found.\n");
    }

    return 0;
}

Node* createNode(Node* parent, Position position) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->parent = parent;
    node->position = position;
    node->g = 0;
    node->h = 0;
    node->f = 0;
    return node;
}

int isEqual(Position pos1, Position pos2) {
    return pos1.row == pos2.row && pos1.col == pos2.col;
}

Position* createPosition(int row, int col) {
    Position* position = (Position*)malloc(sizeof(Position));
    position->row = row;
    position->col = col;
    return position;
}

Position* createPositionArray(int row, int col) {
    Position* positionArray = (Position*)malloc(row * col * sizeof(Position));
    int i, j, index = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            Position position = { i, j };
            positionArray[index++] = position;
        }
    }
    return positionArray;
}

int isWalkable(int maze[MAX_ROWS][MAX_COLS], int row, int col) {
    return maze[row][col] == 0;
}

int getHeuristic(Position pos1, Position pos2) {
    int dx = abs(pos1.row - pos2.row);
    int dy = abs(pos1.col - pos2.col);
    return dx * dx + dy * dy;
}

Position* getAdjacentPositions(Position position) {
    Position* adjacentPositions = (Position*)malloc(8 * sizeof(Position));
    adjacentPositions[0] = *createPosition(position.row, position.col - 1);
    adjacentPositions[1] = *createPosition(position.row, position.col + 1);
    adjacentPositions[2] = *createPosition(position.row - 1, position.col);
    adjacentPositions[3] = *createPosition(position.row + 1, position.col);
    adjacentPositions[4] = *createPosition(position.row - 1, position.col - 1);
    adjacentPositions[5] = *createPosition(position.row - 1, position.col + 1);
    adjacentPositions[6] = *createPosition(position.row + 1, position.col - 1);
    adjacentPositions[7] = *createPosition(position.row + 1, position.col + 1);
    return adjacentPositions;
}

int isPositionInRange(Position position, int row, int col) {
    return position.row >= 0 && position.row < row && position.col >= 0 && position.col < col;
}

Position* filterWalkablePositions(int maze[MAX_ROWS][MAX_COLS], Position* positions, int count, int row, int col) {
    Position* filteredPositions = (Position*)malloc(count * sizeof(Position));
    int i, j, index = 0;
    for (i = 0; i < count; i++) {
        Position position = positions[i];
        if (isPositionInRange(position, row, col) && isWalkable(maze, position.row, position.col)) {
            filteredPositions[index++] = position;
        }
    }
    return filteredPositions;
}

Position* appendPositions(Position* positions1, int count1, Position* positions2, int count2) {
    Position* appendedPositions = (Position*)malloc((count1 + count2) * sizeof(Position));
    int i, index = 0;
    for (i = 0; i < count1; i++) {
        appendedPositions[index++] = positions1[i];
    }
    for (i = 0; i < count2; i++) {
        appendedPositions[index++] = positions2[i];
    }
    return appendedPositions;
}

int isInList(Position position, Position* positions, int count) {
    int i;
    for (i = 0; i < count; i++) {
        if (isEqual(position, positions[i])) {
            return 1;
        }
    }
    return 0;
}

void removeFromList(Position position, Position* positions, int* count) {
    int i, j;
    for (i = 0; i < *count; i++) {
        if (isEqual(position, positions[i])) {
            for (j = i; j < *count - 1; j++) {
                positions[j] = positions[j + 1];
            }
            (*count)--;
            break;
        }
    }
}

Position* astar(int maze[MAX_ROWS][MAX_COLS], int row, int col, Position start, Position end) {
    Node* startNode = createNode(NULL, start);
    startNode->g = startNode->h = startNode->f = 0;
    Node* endNode = createNode(NULL, end);
    endNode->g = endNode->h = endNode->f = 0;

    Position* openList = (Position*)malloc(row * col * sizeof(Position));
    int openListCount = 0;
    Position* closedList = (Position*)malloc(row * col * sizeof(Position));
    int closedListCount = 0;

    openList[openListCount++] = startNode->position;

    while (openListCount > 0) {
        Position current = openList[0];
        int currentIdx = 0;
        int i;
        for (i = 0; i < openListCount; i++) {
            if (getHeuristic(current, endNode->position) < getHeuristic(openList[i], endNode->position)) {
                current = openList[i];
                currentIdx = i;
            }
        }

        removeFromList(current, openList, &openListCount);
        closedList[closedListCount++] = current;

        if (isEqual(current, endNode->position)) {
            Position* path = (Position*)malloc(row * col * sizeof(Position));
            int pathCount = 0;
            Node* current = createNode(NULL, current);
            while (current != NULL) {
                path[pathCount++] = current->position;
                current = current->parent;
            }
            return path;
        }

        Position* adjacentPositions = getAdjacentPositions(current);
        int adjacentCount = 8;
        Position* filteredPositions = filterWalkablePositions(maze, adjacentPositions, adjacentCount, row, col);
        free(adjacentPositions);

        Position* children = appendPositions(filteredPositions, adjacentCount, closedList, closedListCount);
        free(filteredPositions);

        for (i = 0; i < adjacentCount; i++) {
            Position child = children[i];

            if (isInList(child, closedList, closedListCount)) {
                continue;
            }

            Node* childNode = createNode(NULL, child);
            childNode->g = startNode->g + 1;
            childNode->h = getHeuristic(child, endNode->position);
            childNode->f = childNode->g + childNode->h;

            if (isInList(child, openList, openListCount) && childNode->g > startNode->g) {
                continue;
            }

            if (!isInList(child, openList, openListCount)) {
                openList[openListCount++] = child;
            }
        }

        free(children);
    }

    return NULL;
}

void printPath(Position* path, int count) {
    int i;
    for (i = 0; i < count; i++) {
        printf("(%d, %d) ", path[i].row, path[i].col);
    }
    printf("\n");
}
