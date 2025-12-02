# Ant Pheromone Simulation

*GPU-accelerated, multi-agent emergent behavior using compute shaders*
---
[![Ant Sim](https://img.youtube.com/vi/3XrAkhGw1iI/0.jpg)](https://www.youtube.com/watch?v=3XrAkhGw1iI)

---

## Overview

This project is an experimental, GPU-accelerated **multi-agent slime/ant simulation**.
Thousands of agents move across a 2D environment, depositing and following pheromones that diffuse and decay over time.
Simple rules produce surprisingly complex emergent patterns — trails, vortices, branching structures, and swarm behaviors.

Inspired by biological systems like slime molds, ants, and chemotaxis, this simulation explores:

* decentralized computation
* emergent collective intelligence
* shader-based simulation pipelines
* multi-species interactions

---

## Key Features

* **Compute shader agent updates** (WGSL)
* **GPU-driven pheromone diffusion** with adjustable decay + blur kernels
* **Multi-species agents**, each with:
  * steering biases
  * sensor angles
  * trail weights
  * unique pheromone layer
* **Avoid + follow behaviors**, per species
* **Ping-pong textures** to track agent and pheromone state across frames
* Real-time rendering with **Rust + Bevy**

---

## Technical Highlights

### **Shader Pipeline**

* Agents are updated in a compute shader using:

  * vector math
  * sampling of pheromone textures
  * randomized steering
* Pheromone diffusion occurs in a second pass using separable convolution kernels.

### **Data Layout**

* Each agent stores:

  * `position`
  * `angle`
  * `species_index`
* Species settings stored in GPU buffers for different behaviors.
* Pheromones stored in floating-point RGBA texture layers.

### **Emergent Behavior**

* Complex global patterns emerge from:

  * sensor angle offsets
  * turn speeds
  * species-specific pheromone response
  * diffusion strength
  * randomization and noise kernels

You can tune parameters live to alter the ecosystem dynamics.

---

## Running the Project

**Requirements:**

* Rust
* Cargo
* Bevy (version 0.17)

```bash
cargo run --release
```

## Roadmap / Future Work

* Obstacles & environmental constraints
* Interactive parameter control

---

## License

MIT
