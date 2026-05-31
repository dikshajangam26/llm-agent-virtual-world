# 🤖 LLM‑Driven Navigation Agent in a Webots 3D World

This project implements an intelligent agent that navigates a Webots environment using:
- LLM‑based goal interpretation  
- A* path planning  
- Occupancy grid mapping  
- Waypoint following  
- Automatic visualization  
- CSV behaviour logging  

The robot follows natural‑language instructions such as:  
**“Go to the yellow goal, then the red goal, then the green goal.”**

## **Full Documentation:**  
For a complete explanation of the system, including architecture, design choices, planning logic, and implementation details, please refer to the attached **Full Documentation PDF**.

---

## 🚀 Features

- **LLM Multi‑Path Reasoning**  
  Generates multiple candidate goal sequences and selects the best one using an LLM judge.

- ** A* Path Planning **  
  Computes safe paths around inflated obstacles.

- **Waypoint Controller**  
  Uses heading error + distance to follow the planned path.

- **Goal Completion Messages**  
  Prints:  
`[GOAL] red goal reached successfully.`


- **CSV Logging**  
Automatically records robot behaviour into:  `log.csv`


- **Visualization**  
Generates a world map with obstacles, goals, and the selected path:  
`controllers/controller_plane_env/map.png`


---

## 📂 Project Structure



---

## 🛠️ Changing the Route

Modify the natural‑language instruction on **line 430** of:
`controllers/controller_plane_env/controller_plane_env.py`


Example:
"Go to the yellow goal, then the red goal, then the green goal"


---

## ▶️ Running the Simulation

### 1. Install dependencies
```bash
pip install matplotlib requests
```
### 2. Open Webots
* Load plane_env.wbt
* Set controller to controller_plane_env

### 3. Run
* Press Play in Webots.
* Outputs generated:
* map.png → visualization
* log.csv → behaviour log
* Console messages → waypoints + goal completion

---
## Visualization Example

Below is the automatically generated map showing:
- Obstacles  
- Start position  
- Goals  
- Selected path  

<p align="center">
  <img src="Humanoid%20Software/controllers/controller_plane_env/map.png" alt="World Map and Candidate Paths" width="600"/>
</p>

## 🧩 Design Choices

### 1. Why Webots?
Provides a realistic 3D environment with physics, sensors, and easy Python integration.

### 2. Why A* Lattice Planning?
Simple, deterministic, and works perfectly for grid‑based navigation tasks.

### 3. Why an Occupancy Grid?
Allows clean obstacle modeling and safe path planning. **Inflation** ensures the robot maintains a safety buffer and never clips obstacles.

### 4. Why a Discrete Action Space?
Keeps the robot's execution logic simple and robust:
* `turn left`
* `turn right`
* `go forward`
* `stop`

### 5. Why LLM Multi‑Path Reasoning?
Adds high-level intelligence and flexibility to the pipeline:
1. Generates multiple candidate routes.
2. Evaluates and judges each option against the environment context.
3. Selects the safest and most efficient path.

### 6. Why Visualization?
Provides immediate visual proof of the system's performance, helping demonstrate:
* World layout and obstacle boundaries.
* Path quality and optimization.
* General correctness of the planning algorithms.
