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
/controllers
  /controller_plane_env
    controller_plane_env.py
    map.png (generated automatically)
README.md
/worlds
  plane_env.wbt
