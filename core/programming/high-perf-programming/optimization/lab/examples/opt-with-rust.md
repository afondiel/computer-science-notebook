# Core optimization algorithms in Rust. 

## Overview

Rust offers safety, performance, and modern features like ownership, pattern matching, and iterators, making it a great choice for these tasks. 

It covers a representative subset from the general list, advanced examples, domain-specific, and industry-specific implementations, adapting them to Rust’s idioms (e.g., using `Vec`, `rand` crate).

To compile these, you’ll need the `rand` crate. Add this to your `Cargo.toml`:
```toml
[dependencies]
rand = "0.8.5"
```

---
## Table of Contents

- [Overview](#overview)
- [General Comprehensive List](#general-comprehensive-list)
    - [Gradient Descent](#1-gradient-descent)
    - [Genetic Algorithm](#2-genetic-algorithm)
- [Advanced Examples](#advanced-examples)
    - [Adam Optimizer](#1-adam-optimizer)
    - [CMA-ES (Simplified)](#2-cma-es-simplified)
- [By Domain](#by-domain)
    - [Computer Science - Dijkstra’s Algorithm](#1-computer-science---dijkstras-algorithm)
    - [Embedded Systems - Hill Climbing](#2-embedded-systems---hill-climbing)
    - [Computer Vision - Levenberg-Marquardt (Simplified)](#3-computer-vision---levenberg-marquardt-simplified)
- [By Industry](#by-industry)
    - [Autonomous Systems - A*](#1-autonomous-systems---a)
    - [Healthcare - Simulated Annealing](#2-healthcare---simulated-annealing)
    - [Manufacturing - Tabu Search](#3-manufacturing---tabu-search)

---
### Notes
- **Rust Features**: Uses ownership, iterators, and pattern matching for safety and clarity.
- **Randomness**: Relies on the `rand` crate for robust random number generation.
- **Performance**: Comparable to C/C++ with zero-cost abstractions.
- **Libraries**: Could be enhanced with crates like `nalgebra` for linear algebra (e.g., CMA-ES) or `priority-queue` for heap operations.

These implementations cover key examples from previous responses. 
---

### General Comprehensive List
#### 1. Gradient Descent
```rust
fn objective_function(x: f64) -> f64 {
    x * x + 2.0 * x + 1.0
}

fn gradient(x: f64) -> f64 {
    2.0 * x + 2.0
}

fn gradient_descent(x_init: f64, learning_rate: f64, max_iter: i32) -> f64 {
    let mut x = x_init;
    for i in 0..max_iter {
        x -= learning_rate * gradient(x);
        println!("Iteration {}: x = {:.4}, f(x) = {:.4}", i, x, objective_function(x));
    }
    x
}

fn main() {
    let x_init = 5.0;
    let learning_rate = 0.1;
    let max_iter = 10;
    let result = gradient_descent(x_init, learning_rate, max_iter);
    println!("Optimized x: {:.4}", result);
}
```

#### 2. Genetic Algorithm
```rust
use rand::Rng;

fn fitness(x: i32) -> f64 {
    -(x as f64).powi(2) + 10.0 * (x as f64)
}

fn genetic_algorithm(pop_size: usize, max_gen: i32, mutation_rate: f64) -> i32 {
    let mut rng = rand::thread_rng();
    let mut population: Vec<i32> = (0..pop_size).map(|_| rng.gen_range(0..11)).collect();

    for g in 0..max_gen {
        let fitness_values: Vec<f64> = population.iter().map(|&x| fitness(x)).collect();
        let total_fitness: f64 = fitness_values.iter().sum();
        let mut new_population = Vec::with_capacity(pop_size);

        for _ in 0..pop_size {
            let r = rng.gen_range(0.0..total_fitness);
            let mut s = 0.0;
            for (i, &f) in fitness_values.iter().enumerate() {
                s += f;
                if s >= r {
                    new_population.push(population[i]);
                    break;
                }
            }
        }

        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen::<f64>() < 0.8 {
                let p1 = new_population[i];
                let p2 = new_population[i + 1];
                new_population[i] = (p1 + p2) / 2;
                new_population[i + 1] = p1 + p2 - new_population[i];
            }
        }

        for ind in new_population.iter_mut() {
            if rng.gen::<f64>() < mutation_rate {
                *ind = rng.gen_range(0..11);
            }
        }

        population = new_population;
        let best = population.iter().max_by(|a, b| fitness(**a).partial_cmp(&fitness(**b)).unwrap()).unwrap();
        println!("Generation {}: Best x = {}, Fitness = {:.2}", g, best, fitness(*best));
    }

    *population.iter().max_by(|a, b| fitness(**a).partial_cmp(&fitness(**b)).unwrap()).unwrap()
}

fn main() {
    let best = genetic_algorithm(20, 50, 0.1);
    println!("Best solution: x = {}, Fitness = {:.2}", best, fitness(best));
}
```

---

### Advanced Examples
#### 1. Adam Optimizer
```rust
use rand::Rng;

fn objective_function(v: &[f64]) -> f64 {
    v[0] * v[0] + v[1] * v[1]
}

fn gradient(v: &[f64]) -> Vec<f64> {
    vec![2.0 * v[0], 2.0 * v[1]]
}

fn adam_optimizer(init: &[f64], lr: f64, beta1: f64, beta2: f64, max_iter: i32) -> Vec<f64> {
    let mut params = init.to_vec();
    let mut m = vec![0.0; 2];
    let mut v = vec![0.0; 2];
    let epsilon = 1e-8;

    for t in 1..=max_iter {
        let g = gradient(&params);
        for i in 0..2 {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - beta1.powi(t));
            let v_hat = v[i] / (1.0 - beta2.powi(t));
            params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
        if t % 10 == 0 {
            println!("Iteration {}: x = {:.4}, y = {:.4}", t, params[0], params[1]);
        }
    }
    params
}

fn main() {
    let init = vec![5.0, 5.0];
    let result = adam_optimizer(&init, 0.001, 0.9, 0.999, 100);
    println!("Optimized: x = {:.4}, y = {:.4}, f(x, y) = {:.4}", result[0], result[1], objective_function(&result));
}
```

#### 2. CMA-ES (Simplified)
```rust
use rand::Rng;

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

fn cma_es(initial_mean: &[f64], sigma: f64, pop_size: usize, max_iter: i32) -> Vec<f64> {
    let n = initial_mean.len();
    let mut mean = initial_mean.to_vec();
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        cov[i][i] = sigma * sigma;
    }
    let mut rng = rand::thread_rng();

    for iteration in 0..max_iter {
        let mut population = Vec::with_capacity(pop_size);
        let mut fitness = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let mut ind = Vec::with_capacity(n);
            for j in 0..n {
                ind.push(mean[j] + rng.gen_range(-1.0..1.0) * sigma);
            }
            population.push(ind);
            fitness.push(sphere(&population.last().unwrap()));
        }
        let elite_size = pop_size / 2;
        let mut indices: Vec<usize> = (0..pop_size).collect();
        indices.sort_by(|a, b| fitness[*a].partial_cmp(&fitness[*b]).unwrap());
        let elite: Vec<&Vec<f64>> = indices[..elite_size].iter().map(|&i| &population[i]).collect();

        for i in 0..n {
            mean[i] = elite.iter().map(|e| e[i]).sum::<f64>() / elite_size as f64;
        }
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = elite.iter().map(|e| (e[i] - mean[i]) * (e[j] - mean[j])).sum::<f64>() / elite_size as f64;
                if i == j {
                    cov[i][j] += 1e-6;
                }
            }
        }
        println!("Iteration {}: Best fitness = {:.4}", iteration, fitness[indices[0]]);
    }
    mean
}

fn main() {
    let initial_mean = vec![5.0, 5.0];
    let result = cma_es(&initial_mean, 0.5, 10, 100);
    println!("Optimized solution: {:?}", result);
}
```

---

### By Domain
#### 1. Computer Science - Dijkstra’s Algorithm
```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn dijkstra(graph: &Vec<Vec<(usize, i32)>>, src: usize) -> Vec<i32> {
    let n = graph.len();
    let mut dist = vec![i32::MAX; n];
    let mut pq = BinaryHeap::new();
    dist[src] = 0;
    pq.push(Reverse((0, src)));

    while let Some(Reverse((d, u))) = pq.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, weight) in &graph[u] {
            if dist[u] != i32::MAX && dist[u] + weight < dist[v] {
                dist[v] = dist[u] + weight;
                pq.push(Reverse((dist[v], v)));
            }
        }
    }
    dist
}

fn main() {
    let graph = vec![
        vec![(1, 4), (2, 8)],
        vec![(2, 4)],
        vec![(3, 2)],
        vec![],
    ];
    let dist = dijkstra(&graph, 0);
    for (i, &d) in dist.iter().enumerate() {
        println!("Distance to {}: {}", i, d);
    }
}
```

#### 2. Embedded Systems - Hill Climbing
```rust
use rand::Rng;

fn power_usage(duty_cycle: f64) -> f64 {
    duty_cycle * 10.0 + (1.0 - duty_cycle) * 2.0
}

fn hill_climbing(init: f64, step: f64, max_iter: i32) -> f64 {
    let mut current = init;
    let mut rng = rand::thread_rng();
    for _ in 0..max_iter {
        let neighbor = current + rng.gen_range(-step..step);
        if neighbor >= 0.0 && neighbor <= 1.0 && power_usage(neighbor) < power_usage(current) {
            current = neighbor;
        }
    }
    current
}

fn main() {
    let best_duty = hill_climbing(0.5, 0.1, 100);
    println!("Optimal duty cycle: {:.4}, Power: {:.4}", best_duty, power_usage(best_duty));
}
```

#### 3. Computer Vision - Levenberg-Marquardt (Simplified)
```rust
struct Circle {
    x: f64,
    y: f64,
    r: f64,
}

fn error(c: &Circle, points: &[(f64, f64)]) -> f64 {
    points.iter().map(|&(px, py)| {
        let dist = ((px - c.x).powi(2) + (py - c.y).powi(2)).sqrt();
        (dist - c.r).powi(2)
    }).sum()
}

fn lm_fit(points: &[(f64, f64)], init: Circle, max_iter: i32) -> Circle {
    let mut c = init;
    let lambda = 0.001;
    for i in 0..max_iter {
        let e = error(&c, points);
        let dx = (error(&Circle { x: c.x + 0.01, y: c.y, r: c.r }, points) - e) / 0.01;
        let dy = (error(&Circle { x: c.x, y: c.y + 0.01, r: c.r }, points) - e) / 0.01;
        let dr = (error(&Circle { x: c.x, y: c.y, r: c.r + 0.01 }, points) - e) / 0.01;
        c.x -= lambda * dx;
        c.y -= lambda * dy;
        c.r -= lambda * dr;
        if i % 10 == 0 {
            println!("Error: {:.4}", e);
        }
    }
    c
}

fn main() {
    let points = vec![(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
    let init = Circle { x: 0.0, y: 0.0, r: 1.0 };
    let result = lm_fit(&points, init, 50);
    println!("Circle: x = {:.4}, y = {:.4}, r = {:.4}", result.x, result.y, result.r);
}
```

---

### By Industry
#### 1. Autonomous Systems - A*
```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

#[derive(Eq, PartialEq)]
struct Node {
    x: i32,
    y: i32,
    f: f64,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.f.partial_cmp(&other.f).unwrap().reverse()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn a_star(grid: &[[i32; 3]; 3], start: (i32, i32), goal: (i32, i32)) -> Vec<(i32, i32)> {
    let mut visited = [[false; 3]; 3];
    let mut pq = BinaryHeap::new();
    let mut path = Vec::new();
    let h = |x, y| (((x - goal.0).pow(2) + (y - goal.1).pow(2)) as f64).sqrt();
    pq.push(Node { x: start.0, y: start.1, f: h(start.0, start.1) });

    while let Some(current) = pq.pop() {
        if (current.x, current.y) == goal {
            path.push> (current.x, current.y);
            break;
        }
        if visited[current.x as usize][current.y as usize] {
            continue;
        }
        visited[current.x as usize][current.y as usize] = true;
        path.push((current.x, current.y));

        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (dx, dy) in directions {
            let nx = current.x + dx;
            let ny = current.y + dy;
            if nx >= 0 && nx < 3 && ny >= 0 && ny < 3 && grid[nx as usize][ny as usize] == 0 && !visited[nx as usize][ny as usize] {
                let g = path.len() as f64 + 1.0;
                let h = h(nx, ny);
                pq.push(Node { x: nx, y: ny, f: g + h });
            }
        }
    }
    path
}

fn main() {
    let grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
    let path = a_star(&grid, (0, 0), (2, 2));
    for node in path {
        println!("({:?}, {:?})", node.0, node.1);
    }
}
```

#### 2. Healthcare - Simulated Annealing
```rust
use rand::Rng;

fn treatment_effect(dose: f64) -> f64 {
    -dose * dose + 20.0 * dose - 50.0
}

fn simulated_annealing(init_dose: f64, max_dose: f64, max_iter: i32) -> f64 {
    let mut current = init_dose;
    let mut best = current;
    let mut temp = 1000.0;
    let mut rng = rand::thread_rng();

    for _ in 0..max_iter {
        let next = current + rng.gen_range(-1.0..1.0);
        if next >= 0.0 && next <= max_dose {
            let delta = treatment_effect(next) - treatment_effect(current);
            if delta > 0.0 || (-delta / temp).exp() > rng.gen::<f64>() {
                current = next;
            }
            if treatment_effect(current) > treatment_effect(best) {
                best = current;
            }
        }
        temp *= 0.99;
    }
    best
}

fn main() {
    let best_dose = simulated_annealing(5.0, 20.0, 1000);
    println!("Optimal dose: {:.4}, Effect: {:.4}", best_dose, treatment_effect(best_dose));
}
```

#### 3. Manufacturing - Tabu Search
```rust
use rand::Rng;

fn completion_time(schedule: &[usize], times: &[[i32; 2]]) -> i32 {
    let n = schedule.len();
    let mut machine_end = [0; 2];
    for &job in schedule {
        let mut start = 0;
        for i in 0..2 {
            start = start.max(machine_end[i]);
            machine_end[i] = start + times[job][i];
        }
    }
    machine_end[0].max(machine_end[1])
}

fn tabu_search(times: &[[i32; 2]], max_iter: i32) -> Vec<usize> {
    let n = times.len();
    let mut current: Vec<usize> = (0..n).collect();
    let mut best = current.clone();
    let mut tabu_list = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..max_iter {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i != j {
            current.swap(i, j);
            if !tabu_list.contains(&current) {
                let cost = completion_time(&current, times);
                if cost < completion_time(&best, times) {
                    best = current.clone();
                }
                tabu_list.push(current.clone());
                if tabu_list.len() > 10 {
                    tabu_list.remove(0);
                }
            } else {
                current.swap(i, j); // Revert
            }
        }
    }
    best
}

fn main() {
    let times = [[2, 3], [1, 2], [3, 1]];
    let schedule = tabu_search(&times, 100);
    println!("Schedule: {:?}", schedule);
    println!("Completion time: {}", completion_time(&schedule, &times));
}
```

---

