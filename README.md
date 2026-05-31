# LLM‑Driven Navigation Agent in a Webots 3D World  
### Intelligent Goal Selection • A* Path Planning • Receding‑Horizon LLM Control • Visualization • CSV Logging
---
### 📘 Recommended: Full Technical Documentation
A comprehensive description of the system — including design choices, implementation details, and analysis — is available in the attached **Full_Document.pdf**.  
Readers seeking deeper insight should refer to this document.

---
This project implements a fully autonomous navigation agent in Webots using a hybrid of:

- Large Language Model (LLM) reasoning  
- Multi‑path candidate generation + LLM judging  
- Receding‑horizon re‑planning at each goal  
- A* path planning on an occupancy grid  
- Waypoint‑based control  
- Automatic visualization  
- CSV logging for analysis  

The robot receives a natural‑language instruction (e.g., *“Choose the most efficient route to visit all goals”*) and autonomously determines the best route, plans collision‑free paths, and executes them.

---

## 🚀 Features

### **1. Multi‑Path LLM Goal Reasoning**
The system generates multiple candidate goal sequences using different style constraints:

- Shortest distance  
- Clockwise order  
- Counter‑clockwise order  
- Safety‑focused  
- Simplicity‑focused  

A judge LLM scores each candidate on:

- Faithfulness to the instruction  
- Efficiency  
- Reasonableness  
- Clarity  

The highest‑scoring candidate is selected.

---

### **2. Receding‑Horizon LLM Re‑Planning**
Instead of deciding the full route once, the robot:

1. Moves to the next goal  
2. Re‑observes its current position  
3. Updates the list of remaining goals  
4. Calls the LLM again  
5. Selects the next goal dynamically  

This makes the system:

- More robust  
- Less sensitive to initial LLM errors  
- Adaptive at each goal transition  

---

### **3. Snapping LLM Output to Real Goals**
LLMs sometimes hallucinate coordinates.

To prevent invalid or unreachable targets:

- The system snaps the LLM’s chosen coordinate to the **nearest remaining real goal**.
- This guarantees:
  - No hallucinated coordinates  
  - No unreachable goals  
  - No A* failures  
  - Stable behaviour  

This is a major reliability improvement.

---

### **4. A* Path Planning on an Occupancy Grid**
- Obstacles are inflated for safety  
- Grid resolution: `0.20 m`  
- A* computes a collision‑free path  
- Path is converted into world‑space waypoints  

---

### **5. Waypoint‑Based Controller**
At each step:

- Compute distance + heading error  
- Choose one of:
  - `GO_FORWARD`
  - `TURN_LEFT`
  - `TURN_RIGHT`
  - `STOP`

A speed profile slows the robot near goals for stability.

---

### **6. LLM Retry Logic**
Groq API rate‑limits can cause 429 errors.

The controller now includes:

- Automatic retry  
- Exponential backoff  
- Graceful recovery  

This prevents mid‑run crashes.

---

### **7. CSV Logging**
Every control step logs:

- `x, y` position  
- `yaw`  
- `waypoint_index`  
- `action`  
- `distance_to_waypoint`  
- `heading_error`  

Saved as `log.csv` in the project root.

---

### **8. Visualization**
At the end of the run, the system generates `map.png` showing:

- Obstacles  
- Goals (red, green, yellow)  
- Start position  
- Executed path  

---

## Project Structure
```text
Humanoid Software/
│
├── controllers/
│   ├── controller_plane_env/
│   │   ├── controller_plane_env.py      # Main controller (LLM reasoning + planning + control)
│   │   └── map.png                      # Auto‑generated visualization of world + path
│   └── log.csv                          # Auto‑generated behaviour log
│
├── worlds/
│   └── plane_env.wbt                    # Webots world file
│
├── Full_Documentation.pdf               # Complete technical documentation
├── Simulation_video                     # Webot simulation video along with console output
├── README.md                            # Short overview + run instructions
├── LICENSE
└── .gitignore
```

---


---

## 🧠 How It Works (Pipeline)

1. Read user instruction  
2. Generate 4 LLM candidate routes  
3. Judge model scores each  
4. Select best candidate  
5. Snap LLM coordinates to nearest real goal  
6. Build occupancy grid  
7. A* path planning  
8. Follow waypoints  
9. When a goal is reached → re‑plan using LLM  
10. Log + visualize  

---

## 🛠 Installation

### **Python Dependencies**


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


## Design Choices

### **1. Multi‑Path LLM Reasoning**
The system generates several candidate goal sequences using different style constraints (shortest path, clockwise, safety, simplicity).  
A judge LLM scores each candidate, ensuring robust and reliable goal selection.

### **2. Receding‑Horizon LLM Re‑Planning**
After reaching each goal, the robot re‑evaluates:
- its current position  
- remaining goals  
- a fresh LLM decision  

This makes the navigation adaptive and more stable across runs.

### **3. Snapping LLM Output to Real Goals**
LLMs sometimes produce slightly incorrect or hallucinated coordinates.  
To guarantee correctness, the chosen coordinate is always snapped to the **nearest remaining real goal**, preventing unreachable targets and A* failures.

### **4. A* Path Planning**
A classical A* planner runs on an inflated occupancy grid to compute safe, deterministic paths.  
This ensures collision‑free navigation around static obstacles.

### **5. Waypoint‑Based Controller**
The A* path is converted into waypoints.  
A simple discrete controller (forward, turn left, turn right, stop) follows the path using GPS + IMU feedback.

### **6. Retry Logic for LLM Calls**
To handle API rate limits (429 errors), the controller includes automatic retry with backoff, preventing mid‑run crashes.

### **7. Logging and Visualization**
Every step is logged to `log.csv`, and a final `map.png` shows obstacles, goals, and the executed path for easy debugging and analysis.
