# LLM‑Driven Navigation Agent in a Virtual World  
### Intern Challenge – Intelligent Agent Harness in Webots

This project implements a complete intelligent agent system where a Large Language Model (LLM) controls a robot inside a virtual Webots world.  
The agent perceives its environment, reasons about goals, plans paths, and executes actions to achieve tasks such as:

**“Go to the red goal, then the green goal.”**

The system integrates:
- Webots (3D simulation)
- Python controller
- A* lattice path planning
- Occupancy grid mapping
- LLM‑based goal reasoning and selection
- Low‑level waypoint following
- Full visualization of world + obstacles + path

---

# 🚀 Features

### ✔ Virtual environment  
A 3D Webots world containing:
- A TurtleBot3 robot  
- Multiple cardboard‑box obstacles  
- Red and green goal objects  
- A continuous x–y coordinate plane  

### ✔ Observation format  
The agent receives:
- GPS position (x, y)  
- IMU yaw  
- Occupancy grid (static obstacles)  
- LLM‑selected goal sequence  
- Current waypoint target  

### ✔ Action space  
The robot uses a discrete action set:
- `GO_FORWARD`  
- `TURN_LEFT`  
- `TURN_RIGHT`  
- `STOP`

Mapped to differential wheel velocities.

### ✔ LLM‑guided reasoning  
The LLM:
1. Generates multiple candidate goal sequences  
2. Judges each candidate  
3. Selects the best one  
4. Returns structured JSON  
5. Robot executes the chosen plan  

### ✔ Path planning  
- A* lattice planner  
- 0.20 m grid resolution  
- Obstacle inflation for safety  
- Safe stopping radius near goals  
- Smooth waypoint following  

### ✔ Visualization  
After completing the task, the system automatically generates:

- Obstacles  
- Start position  
- Goals  
- Candidate paths  
- Selected path  
- Grid  
- Labels  

Saved as `map.png` inside the controller folder.

---

# 📂 Project Structure
```text
├── controllers/
│   └── controller_plane_env/
│       ├── controller_plane_env.py  # Main robot controller script
│       └── map.png                 # Generated visualization (auto-created)
├── worlds/
│   └── plane_env.wbt               # Webots 3D world environment
└── README.md                       # Project documentation
```

---

# 🧠 System Architecture

### 1. **LLM Goal Reasoning**
The LLM receives:
- User instruction  
- List of objects in the world  

It returns:
```json
{
  "goals": [
    {"x": ..., "y": ...},
    {"x": ..., "y": ...}
  ]
}
```
### 2. Occupancy Grid
Translates physical world barriers into a discrete map for the pathfinder.
* Converts static 3D obstacles into axis-aligned 2D rectangles.
* Applies **inflated safety zones** around obstacles to account for the robot's physical radius.
* Marks corresponding cells in the grid as occupied.

### 3. A* Path Planner
Calculates the shortest collision-free route.
* Transforms world coordinates into grid coordinates.
* Executes the **A\*** search algorithm to find the optimal path.
* Converts grid paths back into world coordinates, producing a sequential list of waypoints.

### 4. Waypoint Follower
A closed-loop controller that drives the robot along the planned path. For each waypoint, it:
* Computes the current distance and heading error.
* Selects the best discrete steering action.
* Translates actions into explicit left/right **wheel velocities** for the Webots differential drive motors.

### 5. Visualization
Generates a real-time `matplotlib` plot tracking the system's internal state:
* Obstacle boundaries and inflation zones.
* Robot start position and final goal targets.
* Candidate paths evaluated vs. the final selected path.

---

## ▶️ How to Run

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install matplotlib requests
```

### 2. Open Webots
1. Launch **Webots**.
2. Open the world file: `worlds/plane_env.wbt`.
3. Verify that the robot's controller is assigned to `controller_plane_env`.

### 3. Run the Simulation
Click the **Play** button in Webots. The console will output live telemetry logs:
```
[LLM] Generating multi-path goal candidates...
[PLANNER] Occupied cells: 60
[SYSTEM] Waypoint 0 reached.
...
[SYSTEM] All waypoints completed.
Saved visualization to: .../controllers/controller_plane_env/map.png
```

### 4. View Results
Open the automatically generated layout image to review the planned path:`controllers/controller_plane_env/map.png`

---

## 📝 Example Input / Output

* **User Input:** *"Go to the red goal, then the green goal."*
* **LLM JSON Response:**
```json
{
  "goals": [
    {"x": 0.19, "y": 1.24},
    {"x": 1.15, "y": -1.26}
  ]
}
```
* **Controller Console Output:**
```
[SYSTEM] Waypoint 19 reached.
[SYSTEM] All waypoints completed.
Saved visualization to: map.png
```

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
