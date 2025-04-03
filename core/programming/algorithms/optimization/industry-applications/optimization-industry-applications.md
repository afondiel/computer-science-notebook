# Optimization algorithms Tailored for Real-World Applications

## Table of Contents

1. [Autonomous Systems](#1-autonomous-systems)
2. [Agriculture](#2-agriculture)
3. [Aerospace/Defense](#3-aerospacedefense)
4. [Healthcare](#4-healthcare)
5. [Smart Cities](#5-smart-cities)
6. [Retail](#6-retail)
7. [Manufacturing](#7-manufacturing)
8. [Summary by Industry](#summary-by-industry)

## 1. **Autonomous Systems**
### Context
Optimization ensures efficient navigation, decision-making, and resource use in drones, self-driving cars, and robots.

### Key Algorithms
- **A* Search**: Path planning.
- **Particle Swarm Optimization (PSO)**: Trajectory optimization.
- **Dynamic Programming**: Motion planning with constraints.
- **Gradient Descent**: Sensor fusion and control tuning.

### Application
Optimizing a drone’s flight path to minimize energy consumption.

### C++ Example (A* for Path Planning)
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

struct Node {
    int x, y;
    double g, h;
    Node(int x_, int y_, double g_, double h_) : x(x_), y(y_), g(g_), h(h_) {}
    double f() const { return g + h; }
    bool operator>(const Node& other) const { return f() > other.f(); }
};

std::vector<Node> a_star(const std::vector<std::vector<int>>& grid, Node start, Node goal) {
    int rows = grid.size(), cols = grid[0].size();
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    std::vector<Node> path;

    pq.push(start);
    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();
        if (current.x == goal.x && current.y == goal.y) {
            path.push_back(current);
            break;
        }
        if (visited[current.x][current.y]) continue;
        visited[current.x][current.y] = true;
        path.push_back(current);

        int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};
        for (int i = 0; i < 4; ++i) {
            int nx = current.x + dx[i], ny = current.y + dy[i];
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && !grid[nx][ny] && !visited[nx][ny]) {
                double h = std::sqrt((nx - goal.x) * (nx - goal.x) + (ny - goal.y) * (ny - goal.y));
                pq.push(Node(nx, ny, current.g + 1, h));
            }
        }
    }
    return path;
}

int main() {
    std::vector<std::vector<int>> grid = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}; // 0 = free, 1 = obstacle
    Node start(0, 0, 0, 0), goal(2, 2, 0, 0);
    auto path = a_star(grid, start, goal);
    for (const auto& node : path) std::cout << "(" << node.x << ", " << node.y << ")\n";
    return 0;
}
```
**Use Case**: Drone navigation avoiding obstacles.

---

## 2. **Agriculture**
### Context
Optimization improves crop yield, resource allocation (water, fertilizer), and automated farming equipment.

### Key Algorithms
- **Linear Programming**: Resource allocation.
- **Genetic Algorithms**: Crop breeding optimization.
- **Simulated Annealing**: Irrigation scheduling.
- **Bayesian Optimization**: Tuning precision agriculture models.

### Application
Optimizing water usage for irrigation.

### C++ Example (Linear Programming - Simplified)
```cpp
#include <iostream>
#include <vector>

double maximize_yield(const std::vector<double>& water, const std::vector<double>& yield_per_unit) {
    double total_yield = 0;
    for (size_t i = 0; i < water.size(); ++i) total_yield += water[i] * yield_per_unit[i];
    return total_yield;
}

std::vector<double> optimize_water(double total_water, const std::vector<double>& yield_per_unit, int max_iter) {
    int n = yield_per_unit.size();
    std::vector<double> water(n, total_water / n); // Initial equal distribution
    double step = 0.1;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double new_water = water[i] + step;
            double old_water = water[i];
            water[i] = new_water;
            double total = 0;
            for (double w : water) total += w;
            if (total > total_water) {
                water[i] = old_water; // Revert if over limit
                continue;
            }
            if (maximize_yield(water, yield_per_unit) < maximize_yield(std::vector<double>(water.begin(), water.begin() + n - 1), yield_per_unit)) {
                water[i] = old_water;
            }
        }
    }
    return water;
}

int main() {
    std::vector<double> yield_per_unit = {5, 3, 4}; // Yield per unit water for 3 crops
    double total_water = 10.0;
    auto water_dist = optimize_water(total_water, yield_per_unit, 100);
    for (double w : water_dist) std::cout << "Water: " << w << "\n";
    std::cout << "Total Yield: " << maximize_yield(water_dist, yield_per_unit) << "\n";
    return 0;
}
```
**Use Case**: Distributing water across crops for maximum yield.

---

## 3. **Aerospace/Defense**
### Context
Optimization enhances aerodynamics, mission planning, and resource efficiency in aircraft, satellites, and defense systems.

### Key Algorithms
- **Gradient Descent**: Control system tuning.
- **Particle Swarm Optimization**: Antenna design.
- **Branch and Bound**: Mission scheduling.
- **CMA-ES**: Structural optimization.

### Application
Optimizing satellite orbit parameters.

### C++ Example (PSO for Orbit Optimization)
```cpp
#include <iostream>
#include <vector>
#include <random>

double fuel_cost(double altitude) {
    return altitude * 0.1 + 1000 / altitude; // Simplified fuel model
}

double pso_orbit(double min_alt, double max_alt, int n_particles, int max_iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_alt, max_alt);
    std::uniform_real_distribution<> dis_v(-1, 1);

    std::vector<double> positions(n_particles), velocities(n_particles, 0), p_best(n_particles);
    double g_best = min_alt;

    for (int i = 0; i < n_particles; ++i) {
        positions[i] = dis(gen);
        p_best[i] = fuel_cost(positions[i]);
        if (p_best[i] < fuel_cost(g_best)) g_best = positions[i];
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n_particles; ++i) {
            velocities[i] += 1 * dis_v(gen) * (p_best[i] - positions[i]) + 2 * dis_v(gen) * (g_best - positions[i]);
            positions[i] = std::max(min_alt, std::min(max_alt, positions[i] + velocities[i]));
            double fitness = fuel_cost(positions[i]);
            if (fitness < p_best[i]) p_best[i] = fitness;
            if (fitness < fuel_cost(g_best)) g_best = positions[i];
        }
    }
    return g_best;
}

int main() {
    double best_altitude = pso_orbit(100, 1000, 20, 50);
    std::cout << "Optimal altitude: " << best_altitude << ", Fuel cost: " << fuel_cost(best_altitude) << "\n";
    return 0;
}
```
**Use Case**: Minimizing fuel for satellite insertion.

---

## 4. **Healthcare**
### Context
Optimization improves medical imaging, treatment planning, and resource allocation.

### Key Algorithms
- **Levenberg-Marquardt**: Medical image registration.
- **Genetic Algorithms**: Drug dosage optimization.
- **Simulated Annealing**: Hospital scheduling.
- **Bayesian Optimization**: Clinical trial design.

### Application
Optimizing radiation therapy dosage.

### C++ Example (Simulated Annealing for Dosage)
```cpp
#include <iostream>
#include <random>

double treatment_effect(double dose) {
    return -dose * dose + 20 * dose - 50; // Simplified efficacy model
}

double simulated_annealing(double init_dose, double max_dose, int max_iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    double current = init_dose, best = current, temp = 1000;

    for (int i = 0; i < max_iter; ++i) {
        double next = current + dis(gen);
        if (next < 0 || next > max_dose) continue;
        double delta = treatment_effect(next) - treatment_effect(current);
        if (delta > 0 || std::exp(delta / temp) > dis(gen)) current = next;
        if (treatment_effect(current) > treatment_effect(best)) best = current;
        temp *= 0.99; // Cooling schedule
    }
    return best;
}

int main() {
    double best_dose = simulated_annealing(5, 20, 1000);
    std::cout << "Optimal dose: " << best_dose << ", Effect: " << treatment_effect(best_dose) << "\n";
    return 0;
}
```
**Use Case**: Balancing efficacy and side effects.

---

## 5. **Smart Cities**
### Context
Optimization enhances traffic flow, energy distribution, and urban planning.

### Key Algorithms
- **Ant Colony Optimization (ACO)**: Traffic routing.
- **Linear Programming**: Energy grid optimization.
- **Genetic Algorithms**: Urban layout planning.
- **Gradient Descent**: Smart sensor calibration.

### Application
Optimizing traffic light timings.

### C++ Example (ACO for Traffic Flow - Simplified)
```cpp
#include <iostream>
#include <vector>
#include <random>

double traffic_cost(const std::vector<int>& path, const std::vector<std::vector<double>>& costs) {
    double total = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) total += costs[path[i]][path[i + 1]];
    return total;
}

std::vector<int> aco_traffic(const std::vector<std::vector<double>>& costs, int n_ants, int max_iter) {
    int n = costs.size();
    std::vector<std::vector<double>> pheromones(n, std::vector<double>(n, 1.0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    std::vector<int> best_path;
    double best_cost = std::numeric_limits<double>::max();

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int ant = 0; ant < n_ants; ++ant) {
            std::vector<int> path = {0};
            std::vector<bool> visited(n, false);
            visited[0] = true;
            for (int i = 1; i < n; ++i) {
                double total = 0;
                std::vector<double> probs(n);
                for (int j = 0; j < n; ++j) {
                    if (!visited[j]) total += pheromones[path.back()][j] / costs[path.back()][j];
                }
                double r = dis(gen) * total;
                double sum = 0;
                for (int j = 0; j < n; ++j) {
                    if (!visited[j]) {
                        sum += pheromones[path.back()][j] / costs[path.back()][j];
                        if (sum >= r) {
                            path.push_back(j);
                            visited[j] = true;
                            break;
                        }
                    }
                }
            }
            double cost = traffic_cost(path, costs);
            if (cost < best_cost) {
                best_cost = cost;
                best_path = path;
            }
            for (size_t i = 0; i < path.size() - 1; ++i) pheromones[path[i]][path[i + 1]] += 1 / cost;
        }
    }
    return best_path;
}

int main() {
    std::vector<std::vector<double>> costs = {{0, 4, 8}, {4, 0, 2}, {8, 2, 0}};
    auto path = aco_traffic(costs, 10, 50);
    for (int node : path) std::cout << node << " ";
    std::cout << "\nCost: " << traffic_cost(path, costs) << "\n";
    return 0;
}
```
**Use Case**: Reducing congestion at intersections.

---

## 6. **Retail**
### Context
Optimization improves inventory management, pricing, and customer targeting.

### Key Algorithms
- **Dynamic Programming**: Inventory optimization.
- **Gradient Descent**: Demand forecasting.
- **Genetic Algorithms**: Store layout optimization.
- **Bayesian Optimization**: Price elasticity tuning.

### Application
Optimizing stock levels for seasonal demand.

### C++ Example (DP for Inventory)
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int inventory_cost(int stock, int demand) {
    return stock < demand ? 10 * (demand - stock) : 2 * (stock - demand); // Penalty for under/overstock
}

std::vector<int> dp_inventory(const std::vector<int>& demands, int max_stock) {
    int n = demands.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(max_stock + 1, 0));
    std::vector<int> policy(n);

    for (int i = n - 1; i >= 0; --i) {
        for (int s = 0; s <= max_stock; ++s) {
            dp[i][s] = inventory_cost(s, demands[i]) + (i + 1 < n ? dp[i + 1][s] : 0);
            if (i == 0) policy[i] = s;
            for (int order = 0; order <= max_stock - s; ++order) {
                int next_stock = s + order;
                int cost = inventory_cost(next_stock, demands[i]) + (i + 1 < n ? dp[i + 1][next_stock] : 0);
                if (cost < dp[i][s]) {
                    dp[i][s] = cost;
                    if (i == 0) policy[i] = next_stock;
                }
            }
        }
    }
    return policy;
}

int main() {
    std::vector<int> demands = {5, 3, 7};
    auto policy = dp_inventory(demands, 10);
    for (int s : policy) std::cout << "Stock: " << s << "\n";
    return 0;
}
```
**Use Case**: Minimizing stockout and overstock costs.

---

## 7. **Manufacturing**
### Context
Optimization streamlines production scheduling, supply chain, and quality control.

### Key Algorithms
- **Simplex Method**: Production planning.
- **Tabu Search**: Job shop scheduling.
- **Genetic Algorithms**: Assembly line balancing.
- **Gradient Descent**: Process parameter tuning.

### Application
Scheduling jobs on machines to minimize completion time.

### C++ Example (Tabu Search for Scheduling)
```cpp
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

double completion_time(const std::vector<int>& schedule, const std::vector<std::vector<int>>& times) {
    int n = schedule.size(), m = times[0].size();
    std::vector<int> machine_end(m, 0);
    for (int job : schedule) {
        int start = 0;
        for (int i = 0; i < m; ++i) {
            start = std::max(start, machine_end[i]);
            machine_end[i] = start + times[job][i];
        }
    }
    return *std::max_element(machine_end.begin(), machine_end.end());
}

std::vector<int> tabu_search(const std::vector<std::vector<int>>& times, int max_iter) {
    int n = times.size();
    std::vector<int> current(n), best(n);
    std::iota(current.begin(), current.end(), 0); // Initial: 0, 1, 2, ...
    best = current;
    std::vector<std::vector<int>> tabu_list;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    for (int iter = 0; iter < max_iter; ++iter) {
        int i = dis(gen), j = dis(gen);
        if (i != j) {
            std::swap(current[i], current[j]);
            if (std::find(tabu_list.begin(), tabu_list.end(), current) == tabu_list.end()) {
                double cost = completion_time(current, times);
                if (cost < completion_time(best, times)) best = current;
                tabu_list.push_back(current);
                if (tabu_list.size() > 10) tabu_list.erase(tabu_list.begin());
            } else {
                std::swap(current[i], current[j]); // Revert
            }
        }
    }
    return best;
}

int main() {
    std::vector<std::vector<int>> times = {{2, 3}, {1, 2}, {3, 1}}; // Jobs x Machines
    auto schedule = tabu_search(times, 100);
    for (int job : schedule) std::cout << job << " ";
    std::cout << "\nCompletion time: " << completion_time(schedule, times) << "\n";
    return 0;
}
```
**Use Case**: Minimizing makespan in a factory.

---

## Summary by Industry
- **Autonomous Systems**: Path and trajectory optimization (e.g., A*).
- **Agriculture**: Resource and yield optimization (e.g., Linear Programming).
- **Aerospace/Defense**: Design and mission efficiency (e.g., PSO).
- **Healthcare**: Treatment and scheduling (e.g., Simulated Annealing).
- **Smart Cities**: Traffic and energy management (e.g., ACO).
- **Retail**: Inventory and pricing (e.g., DP).
- **Manufacturing**: Production and scheduling (e.g., Tabu Search).

These examples can be extended with real-world data, parallelism (e.g., OpenMP), or libraries like Gurobi (for LP) or Eigen (for matrix ops). Let me know if you’d like deeper exploration of any industry!