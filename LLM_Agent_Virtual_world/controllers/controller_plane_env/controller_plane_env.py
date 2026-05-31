from controller import Robot
import math
import json
import requests
import os
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

# ============================================================
# CONFIG & LOGGING
# ============================================================

TIME_STEP = 32

# CSV LOGGING (saved in project root)
LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "log.csv"
)

with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "x", "y", "yaw",
        "waypoint_index",
        "action",
        "distance_to_waypoint",
        "heading_error"
    ])

# LLM CONFIG
GROQ_API_KEY = os.getenv("OPENAI_API_KEY")
if GROQ_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_MAIN = "llama-3.1-8b-instant"
GROQ_MODEL_JUDGE = "llama-3.1-8b-instant"

# WORLD OBJECTS
OBJECTS = [
    {"name": "red goal",    "x": 0.19,  "y": 1.24},
    {"name": "green goal",  "x": 1.15,  "y": -1.26},
    {"name": "yellow goal", "x": -2.25, "y": 0.48}
]

# STATIC OBSTACLES (xmin, xmax, ymin, ymax)
OBSTACLES = [
    (-1.99, -1.39, 1.07, 1.67),
    (-0.655, -0.055, -2.21, -1.61),
    (0.49, 1.09, -0.61, -0.01),
]

OBSTACLE_INFLATION = 0.15  # safety margin

# MOTION CONSTANTS
FORWARD_SPEED = 4.0
TURN_SPEED = 2.0
WAYPOINT_REACHED_DIST = 0.20
GOAL_STOP_RADIUS = 0.30  # stop this far before the exact goal point

# GRID / A*
GRID_RES = 0.20
GRID_RADIUS = 10


# ============================================================
# SPEED PROFILE & CONTROL
# ============================================================

def speed_from_distance(dist):
    FAR_DIST = 1.0
    NEAR_DIST = 0.30

    if dist > FAR_DIST:
        return FORWARD_SPEED
    elif dist > NEAR_DIST:
        alpha = (dist - NEAR_DIST) / (FAR_DIST - NEAR_DIST)
        return (0.7 + 0.3 * alpha) * FORWARD_SPEED
    else:
        return 0.4 * FORWARD_SPEED


def action_to_wheels(action, dist_goal):
    if action == "GO_FORWARD":
        v = speed_from_distance(dist_goal)
        return v, v
    elif action == "TURN_LEFT":
        return -TURN_SPEED, TURN_SPEED
    elif action == "TURN_RIGHT":
        return TURN_SPEED, -TURN_SPEED
    elif action == "STOP":
        return 0.0, 0.0
    else:
        return 0.0, 0.0


def compute_error_xy(gps, imu, target_x, target_y):
    pos = gps.getValues()
    x = pos[0]
    y = pos[1]

    dx = target_x - x
    dy = target_y - y
    dist = math.hypot(dx, dy)

    target_heading = math.atan2(dy, dx)
    roll, pitch, yaw = imu.getRollPitchYaw()

    heading_err = target_heading - yaw
    while heading_err > math.pi:
        heading_err -= 2 * math.pi
    while heading_err < -math.pi:
        heading_err += 2 * math.pi

    return dist, heading_err


def decide_action(dist, heading_err):
    if dist < WAYPOINT_REACHED_DIST:
        return "STOP"

    if abs(heading_err) > 0.25:
        return "TURN_LEFT" if heading_err > 0 else "TURN_RIGHT"

    return "GO_FORWARD"


# ============================================================
# LLM HELPERS – MULTI-PATH + JUDGE (RECEDING HORIZON)
# ============================================================

STYLE_CONSTRAINTS = [
    "Prioritize shortest overall travel distance.",
    "Prioritize visiting goals in a clockwise spatial order.",
    "Prioritize visiting goals in a counter-clockwise spatial order.",
    "Prioritize safety and avoiding tight spaces.",
    "Prioritize simplicity of the route.",
]

GENERATION_TEMPLATE = """
You are an intelligent robot navigation planner in a 2D world.

You are given a set of named objects (goals) with coordinates:
{objects_json}

The robot is currently at position:
{current_position_json}

User instruction:
"{user_instruction}"

You must propose ONE ordered list of goals for the robot to visit next,
expressed as a list of (x, y) coordinates. The first goal in the list
is the next goal the robot should go to.

Additional style constraint for this candidate:
"{style_constraint}"

Return ONLY valid JSON, no extra text:
{{
  "goals": [
    {{"x": number, "y": number}}
  ]
}}
"""

JUDGE_TEMPLATE = """
You are an impartial judge evaluating candidate goal sequences for a robot in a 2D world.

Objects in the world:
{objects_json}

The robot is currently at position:
{current_position_json}

User instruction:
"{user_instruction}"

You are given multiple candidate solutions. Each candidate is a JSON object:
{{
  "goals": [{{"x": number, "y": number}}]
}}

Evaluate each candidate on:
- how well it follows the user instruction
- reasonableness of the route
- efficiency (not absurdly long or random)
- clarity (no nonsense coordinates)

Then:
1. Assign each candidate a score from 1 to 10.
2. Choose the single best candidate.

Return ONLY valid JSON, no extra text:
{{
  "scores": [
    {{
      "index": <int>,  # 0-based index of candidate
      "score": <int>,  # 1-10
      "justification": "string"
    }}
  ],
  "winner_index": <int>  # 0-based index of best candidate
}}
"""


import time

def call_groq_chat(model, messages, temperature=0.2, max_tokens=512, retries=5):
    for attempt in range(retries):
        try:
            resp = requests.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            # Handle rate limit (429)
            if resp.status_code == 429:
                wait = 1.5 * (attempt + 1)
                print(f"[LLM] Rate limited (429). Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                raise e

        except Exception as e:
            print(f"[LLM] Unexpected error: {e}. Retrying...")
            time.sleep(1)

    raise RuntimeError("[LLM] Failed after multiple retries.")


def llm_generate_goal_candidate(user_instruction, objects, current_pos, style_constraint):
    prompt = GENERATION_TEMPLATE.format(
        objects_json=json.dumps(objects, indent=2),
        current_position_json=json.dumps({"x": current_pos[0], "y": current_pos[1]}, indent=2),
        user_instruction=user_instruction,
        style_constraint=style_constraint,
    )
    content = call_groq_chat(
        model=GROQ_MODEL_MAIN,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"goals": [], "raw": content}


def llm_generate_goal_candidates(user_instruction, objects, current_pos, n_candidates=4):
    candidates = []
    for i in range(n_candidates):
        style = STYLE_CONSTRAINTS[i % len(STYLE_CONSTRAINTS)]
        cand = llm_generate_goal_candidate(user_instruction, objects, current_pos, style)
        candidates.append(cand)
    return candidates


def llm_judge_goal_candidates(user_instruction, objects, current_pos, candidates):
    prompt = JUDGE_TEMPLATE.format(
        objects_json=json.dumps(objects, indent=2),
        current_position_json=json.dumps({"x": current_pos[0], "y": current_pos[1]}, indent=2),
        user_instruction=user_instruction,
    ) + "\n\nCandidates:\n" + json.dumps(candidates, indent=2)

    content = call_groq_chat(
        model=GROQ_MODEL_JUDGE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "scores": [
                {"index": 0, "score": 5, "justification": "Judge JSON invalid"}
            ],
            "winner_index": 0,
        }
    winner_idx = result.get("winner_index", 0)
    return winner_idx, result


def solve_goals_with_multipath(user_instruction, objects, current_pos, n_candidates=4):
    candidates = llm_generate_goal_candidates(user_instruction, objects, current_pos, n_candidates)
    winner_idx, judge_result = llm_judge_goal_candidates(user_instruction, objects, current_pos, candidates)
    best = candidates[winner_idx]
    best_goals = best.get("goals", [])
    meta = {
        "candidates": candidates,
        "winner_index": winner_idx,
        "judge_result": judge_result,
    }
    return best_goals, meta


# ============================================================
# PLANNING: OCCUPANCY GRID + A*
# ============================================================

def world_point_in_obstacle(x, y):
    for (xmin, xmax, ymin, ymax) in OBSTACLES:
        if (xmin - OBSTACLE_INFLATION) <= x <= (xmax + OBSTACLE_INFLATION) and \
           (ymin - OBSTACLE_INFLATION) <= y <= (ymax + OBSTACLE_INFLATION):
            return True
    return False


def world_to_grid(x, y):
    return int(round(x / GRID_RES)), int(round(y / GRID_RES))


def grid_to_world(gx, gy):
    return gx * GRID_RES, gy * GRID_RES


def build_occupancy_grid():
    occupied = set()
    max_cells = int(GRID_RADIUS / GRID_RES)
    for gx in range(-max_cells, max_cells + 1):
        for gy in range(-max_cells, max_cells + 1):
            wx, wy = grid_to_world(gx, gy)
            if world_point_in_obstacle(wx, wy):
                occupied.add((gx, gy))
    print(f"[PLANNER] Occupied cells: {len(occupied)}")
    return occupied


def neighbors(gx, gy, occupied):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ngx = gx + dx
        ngy = gy + dy
        if abs(ngx) <= GRID_RADIUS / GRID_RES and abs(ngy) <= GRID_RADIUS / GRID_RES:
            if (ngx, ngy) not in occupied:
                yield ngx, ngy


def heuristic(gx, gy, gx2, gy2):
    return math.hypot(gx2 - gx, gy2 - gy)


def a_star(start_g, goal_g, occupied):
    sx, sy = start_g
    gx, gy = goal_g

    open_set = []
    heapq.heappush(open_set, (0.0, (sx, sy)))
    came_from = {}
    g_score = {(sx, sy): 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for nb in neighbors(*current, occupied):
            tentative = g_score[current] + math.hypot(nb[0] - current[0], nb[1] - current[1])
            if nb not in g_score or tentative < g_score[nb]:
                g_score[nb] = tentative
                f = tentative + heuristic(nb[0], nb[1], gx, gy)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = current

    return None


def plan_path_xy(start_x, start_y, goal_x, goal_y, occupied):
    start_g = world_to_grid(start_x, start_y)
    goal_g = world_to_grid(goal_x, goal_y)
    grid_path = a_star(start_g, goal_g, occupied)
    if grid_path is None:
        print("[PLANNER] No path found.")
        return []
    return [grid_to_world(gx, gy) for gx, gy in grid_path]


# ============================================================
# VISUALIZATION & LOGGING
# ============================================================

def visualize_world_and_paths(start_x, start_y, objects, obstacles, path):
    fig, ax = plt.subplots(figsize=(8, 8))

    for (xmin, xmax, ymin, ymax) in obstacles:
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), w, h,
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.4
        )
        ax.add_patch(rect)

    for obj in objects:
        if obj["name"] == "yellow goal":
            color = "yellow"
        elif obj["name"] == "green goal":
            color = "green"
        elif obj["name"] == "red goal":
            color = "red"
        else:
            color = "black"

        ax.plot(obj["x"], obj["y"], 'o', markersize=10, color=color)
        ax.text(obj["x"] + 0.05, obj["y"] + 0.05, obj["name"], color=color)

    ax.plot(start_x, start_y, 'bo', markersize=12, label="Start")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=3, color='red', label="Executed Path")

    ax.set_title("World Map + Executed Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)

    controller_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(controller_dir, "map.png")
    plt.savefig(save_path, dpi=300)
    print("Saved visualization to:", save_path)


def log_step(gps, imu, wp_index, action, dist, heading_err):
    pos = gps.getValues()
    roll, pitch, yaw = imu.getRollPitchYaw()

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            pos[0], pos[1], yaw,
            wp_index,
            action,
            dist,
            heading_err
        ])


# ============================================================
# MAIN – RECEDING-HORIZON LLM + STATE MACHINE
# ============================================================

def main():
    robot = Robot()

    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)

    imu = robot.getDevice("inertial unit")
    imu.enable(TIME_STEP)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

    # Wait for GPS
    while robot.step(TIME_STEP) != -1:
        pos = gps.getValues()
        if not math.isnan(pos[0]) and not math.isnan(pos[1]):
            break

    start_x = pos[0]
    start_y = pos[1]

    # Build occupancy grid once
    occupied = build_occupancy_grid()

    # Remaining goals (mutable copy)
    remaining_goals = OBJECTS.copy()

    # For visualization: store all executed waypoints
    executed_path = [(start_x, start_y)]

    # State for current segment
    current_segment_waypoints = []
    wp_index = 0
    current_goal_obj = None
    planning_needed = True

    # Base instruction for all LLM calls
    base_instruction = "Choose the most efficient route to visit all remaining goals from the current position."

    while robot.step(TIME_STEP) != -1:
        pos = gps.getValues()
        cur_x, cur_y = pos[0], pos[1]

        # ----------------- PLANNING STATE -----------------
        if planning_needed:
            if not remaining_goals:
                print("[SYSTEM] All goals completed.")
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                visualize_world_and_paths(start_x, start_y, OBJECTS, OBSTACLES, executed_path)
                break

            print("[LLM] Generating multi-path goal candidates for remaining goals...")
            best_goals, meta = solve_goals_with_multipath(
                base_instruction,
                remaining_goals,
                current_pos=(cur_x, cur_y),
                n_candidates=4
            )
            print(f"[LLM] Winner index: {meta['winner_index']}")
            for s in meta["judge_result"]["scores"]:
                print(f"[LLM] Candidate {s['index']}: {s['score']} | {s['justification']}")

            if not best_goals:
                print("[LLM] No valid goals returned, stopping.")
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                visualize_world_and_paths(start_x, start_y, OBJECTS, OBSTACLES, executed_path)
                break

            # Take the first goal in the best sequence as the next goal
            next_goal = best_goals[0]
            gx, gy = next_goal["x"], next_goal["y"]

            # Match to remaining_goals to recover name
            # --- SNAP LLM COORDINATES TO NEAREST REMAINING GOAL ---
            best_obj = None
            best_dist = float("inf")
            
            for obj in remaining_goals:
                d = math.hypot(obj["x"] - gx, obj["y"] - gy)
                if d < best_dist:
                    best_dist = d
                    best_obj = obj
            
            # Always snap to the nearest remaining goal
            if best_obj is None:
                print("[LLM] Could not match any remaining goal, stopping.")
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                visualize_world_and_paths(start_x, start_y, OBJECTS, OBSTACLES, executed_path)
                return
            
            current_goal_obj = best_obj
            gx, gy = current_goal_obj["x"], current_goal_obj["y"]

            # Plan path to a safe point before the goal
            dx = gx - cur_x
            dy = gy - cur_y
            dist = math.hypot(dx, dy)

            if dist > GOAL_STOP_RADIUS:
                scale = (dist - GOAL_STOP_RADIUS) / dist
                safe_goal_x = cur_x + dx * scale
                safe_goal_y = cur_y + dy * scale
            else:
                safe_goal_x = gx
                safe_goal_y = gy

            segment_wps = plan_path_xy(cur_x, cur_y, safe_goal_x, safe_goal_y, occupied)
            if len(segment_wps) <= 1:
                print("[PLANNER] No valid segment path, stopping.")
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                visualize_world_and_paths(start_x, start_y, OBJECTS, OBSTACLES, executed_path)
                break

            current_segment_waypoints = segment_wps[1:]
            wp_index = 0
            planning_needed = False
            print(f"[SYSTEM] New segment planned to {current_goal_obj['name']} with {len(current_segment_waypoints)} waypoints.")

        # ----------------- FOLLOWING STATE -----------------
        if wp_index >= len(current_segment_waypoints):
            # Segment done: goal reached
            print(f"[GOAL] {current_goal_obj['name']} reached successfully.")
            remaining_goals = [g for g in remaining_goals if not (
                abs(g["x"] - current_goal_obj["x"]) < 0.05 and
                abs(g["y"] - current_goal_obj["y"]) < 0.05
            )]
            planning_needed = True
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            continue

        tx, ty = current_segment_waypoints[wp_index]
        dist, heading_err = compute_error_xy(gps, imu, tx, ty)

        if dist < WAYPOINT_REACHED_DIST:
            print(f"[SYSTEM] Waypoint {wp_index} reached.")
            executed_path.append((tx, ty))
            wp_index += 1
            continue

        action = decide_action(dist, heading_err)
        vl, vr = action_to_wheels(action, dist)

        log_step(gps, imu, wp_index, action, dist, heading_err)

        left_motor.setVelocity(vl)
        right_motor.setVelocity(vr)


if __name__ == "__main__":
    main()
