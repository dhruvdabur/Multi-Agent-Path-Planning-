#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from hb_interfaces.msg import Poses2D, BotCmd, BotCmdArray
from linkattacher_msgs.srv import AttachLink, DetachLink
import numpy as np
import math
import time
import random
import itertools
import json
import heapq
from std_msgs.msg import String

ENABLE_LOOP_TRACE_LOGS = True
LOOP_TRACE_LOG_INTERVAL_S = 1.0
# ---------------------- PID Controller Class --------------------------------
class PID:
    def __init__(self, kp, ki, kd, max_out=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_error = 0.0
       

    def compute(self, error, dt):
        if dt <= 0:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return float(np.clip(output, -self.max_out, self.max_out))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# ---------------------- Raster Grid + A* -----------------------------------
class RasterGrid:
    def __init__(self, width_mm, height_mm, resolution_mm):
        self.resolution = resolution_mm
        self.cols = int(width_mm / resolution_mm)
        self.rows = int(height_mm / resolution_mm)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

    def world_to_grid(self, x, y):
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return (max(0, min(row, self.rows - 1)), max(0, min(col, self.cols - 1)))

    def grid_to_world(self, row, col):
        x = (col * self.resolution) + (self.resolution / 2.0)
        y = (row * self.resolution) + (self.resolution / 2.0)
        return (x, y)

    def clear_grid(self):
        self.grid.fill(0)

    def set_obstacle(self, x, y, radius_mm):
        cell_radius = int(radius_mm / self.resolution)
        r_idx, c_idx = self.world_to_grid(x, y)
        r_start = max(0, r_idx - cell_radius)
        r_end = min(self.rows, r_idx + cell_radius + 1)
        c_start = max(0, c_idx - cell_radius)
        c_end = min(self.cols, c_idx + cell_radius + 1)
        self.grid[r_start:r_end, c_start:c_end] = 1


def a_star_search(grid_obj, start_xy, goal_xy):
    start = grid_obj.world_to_grid(start_xy[0], start_xy[1])
    goal = grid_obj.world_to_grid(goal_xy[0], goal_xy[1])

    if grid_obj.grid[start] == 1 or grid_obj.grid[goal] == 1:
        return [], 999999.0

    neighbors = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]
    open_set = [(0.0, start[0], start[1])]
    g_score = np.full((grid_obj.rows, grid_obj.cols), np.inf)
    g_score[start] = 0.0
    came_from = {}

    def get_heuristic(r, c):
        dr = abs(r - goal[0])
        dc = abs(c - goal[1])
        return (max(dr, dc) + 0.414 * min(dr, dc)) * 1.001

    while open_set:
        _, r, c = heapq.heappop(open_set)
        if (r, c) == goal:
            path = []
            path_length_mm = 0.0
            curr = (r, c)
            while curr in came_from:
                path.append(grid_obj.grid_to_world(curr[0], curr[1]))
                prev = came_from[curr]
                p1 = grid_obj.grid_to_world(curr[0], curr[1])
                p2 = grid_obj.grid_to_world(prev[0], prev[1])
                path_length_mm += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                curr = prev
            path.reverse()
            return path, path_length_mm

        for dr, dc, weight in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_obj.rows and 0 <= nc < grid_obj.cols:
                if grid_obj.grid[nr, nc] == 1:
                    continue
                tentative_g = g_score[r, c] + weight
                if tentative_g < g_score[nr, nc]:
                    came_from[(nr, nc)] = (r, c)
                    g_score[nr, nc] = tentative_g
                    f_score = tentative_g + get_heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, nr, nc))

    return [], 999999.0


# ---------------------- Per-bot State ---------------------------------------
class BotState:
    def __init__(self, bot_id, node, pid_params, final_goal, return_goal):
        self.id = bot_id
        self.node = node
        self.pose = None
        self.current_crate = None
        self.tracked_crate_id = None
        self.goal = None
        # [COLLISION AVOIDANCE] State variables
        self.blocked_by_crate = False # <--- ADD THIS LINE
        self.pose_timestamp = node.get_clock().now()

        # state flags
        self.goal_reached = False
        self.arm_placed = False
        self.box_attached = False
        self.arm_lifted = False
        self.move_after_attach = False
        self.box_detached = False
        self.return_to_start = False
        self.going_to_staging_point = False
        self.has_returned_home = False 
        self.is_permanently_idle = False

        self.completed_crates = set()
        self.last_time = node.get_clock().now()

        # [COLLISION AVOIDANCE] State variables
        self.pose_timestamp = node.get_clock().now()
        self.prev_pose_for_vel = None
        self.velocity = 0.0
        self.is_moving = False
        self.slowdown_scale = 1.0
        self.priority_cooldown_end_time = None
        self.loop_state = "INIT"
        self.loop_state_enter_time = time.time()
        self.last_loop_state_log_time = 0.0
        
        # [FOLLOW BEHAVIOR] New variables
        self.is_following = False
        self.follow_target_id = None
        self.follow_offset_distance = 245.0  # Stay this far behind the leader
        self.current_path = []
        self.path_goal_cache = None
        self.last_path_calc_time = 0.0

        # PID controllers
        self.pid_x = PID(**pid_params['x'])
        self.pid_y = PID(**pid_params['y'])
        self.pid_theta = PID(**pid_params['theta'])

        self.orig_pid_limits = {
            'x': pid_params['x']['max_out'],
            'y': pid_params['y']['max_out'],
            'theta': pid_params['theta']['max_out']
        }

        # Goals
        self.final_goal = final_goal.copy()
        self.return_goal = return_goal.copy()

    def reset_pid(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_theta.reset()


# ---------------------- Main Node Class -------------------------------------
class HolonomicMoveToCratesMulti(Node):
    def __init__(self):
        super().__init__('holonomic_move_to_crates_multi')
        self.enable_loop_trace_logs = ENABLE_LOOP_TRACE_LOGS
        self.loop_trace_interval_s = LOOP_TRACE_LOG_INTERVAL_S
        # Visualization Publisher
        self.viz_publisher = self.create_publisher(String, '/zone_debug', 10)
        # Robot Parameters
        self.bot_ids = [0, 2, 4]
        self.current_pose = None
        self.crates = []
        self.drop_positions = {}
        self.drop_positions_assigned = False

        self.last_time = self.get_clock().now()
        self.raster = RasterGrid(2438.4, 2438.4, 20.0)

        self.xy_tolerance = 5.0
        self.theta_tolerance_deg = 3.0
        self.theta_tolerance_center_deg = 15.0
        self.max_vel = 50.0
        
        # [MODIFIED] Collision avoidance parameters
        self.follow_distance_threshold = 250.0  # Distance to trigger follow behavior
        self.crate_avoidance_threshold = 180.0
        self.safety_radius = 550.0
        self.critical_radius = 300.0

        self.final_goal = np.array([1219.0, 1180.0, 0.0])
        self.return_goal = np.array([1218.0, 205.0, 0.0])
        

        # MODIFIED: Staging points for ALL bots
        self.staging_point_0 = np.array([600.0, 1219.0, 0.0])
        self.staging_point_2 = np.array([1700.0, 1200.0, 0.0])
        self.staging_point_4 = np.array([800.0, 1200.0, 0.0])
        
        # Staging threshold
        self.staging_y_threshold = 1400.0
        
        # Idle home points
        self.idle_home_point_0 = np.array([1218.0, 205.0, 0.0])
        self.idle_home_point_2 = np.array([1568.0, 202.0, 0.0])
        self.idle_home_point_4 = np.array([864.0, 204.0, 0.0])

        # Crate region
        self.x_min, self.x_max = 1020, 1410
        self.y_min, self.y_max = 1075, 1355

        # Crate geometry
        self.crate_size = 60.0
        self.safe_gap = 10.0
        self.min_dist = self.crate_size + self.safe_gap

        # PID parameters
        self.pid_params = {
            'x': {'kp': 4, 'ki': 0.0001, 'kd': 0.001, 'max_out': self.max_vel},
            'y': {'kp': 5, 'ki': 0.0001, 'kd': 0.0001, 'max_out': self.max_vel},
            'theta': {'kp': -9, 'ki': 0.001, 'kd': 0.0002, 'max_out': self.max_vel * 2}
        }

        # Per-bot states
        self.bots = {bid: BotState(bid, self, self.pid_params, self.final_goal, self.return_goal) for bid in self.bot_ids}
        if 0 in self.bots:
            self.bots[0].return_goal = self.idle_home_point_0.copy()

        # Drop zones
        self.drop_zones = {
            0: {'x_min': 1020, 'x_max': 1400, 'y_min': 1075, 'y_max': 1355},
            2: {'x_min': 1470, 'x_max': 1752, 'y_min': 1920, 'y_max': 2115},
            1: {'x_min': 675,  'x_max': 920,  'y_min': 1920, 'y_max': 2115}
        }

        # REMOVED: self.zone_locks = {0: None, 1: None, 2: None}
        
        # NEW: Define 3 subparts for each zone [Full_Center, Half_1, Half_2]
        # Format: (x, y)
        self.zone_subparts = {}
        
        # --- Zone 0 (Top) ---
        # Range: X=1020-1400, Y=1075-1355. Split X (Left/Right)
        z0_mid_x = (1020 + 1400) / 2
        z0_y = (1075 + 1355) / 2
        self.zone_subparts[0] = [
            (z0_mid_x, z0_y),           # 0: Full Center
            (1115.0, z0_y),             # 1: Left Half Center
            (1305.0, z0_y)              # 2: Right Half Center
        ]

        # --- Zone 1 (Left) ---
        # Range: X=675-920, Y=1920-2115. Split Y (Top/Bottom)
        z1_x = (675 + 920) / 2
        z1_mid_y = (1920 + 2115) / 2
        self.zone_subparts[1] = [
            (z1_x, z1_mid_y),           # 0: Full Center
            (z1_x, 1968.0),             # 1: Bottom Half
            (z1_x, 2066.0)              # 2: Top Half
        ]

        # --- Zone 2 (Right) ---
        # Range: X=1470-1752, Y=1920-2115. Split Y (Top/Bottom)
        z2_x = (1470 + 1752) / 2
        z2_mid_y = (1920 + 2115) / 2
        self.zone_subparts[2] = [
            (z2_x, z2_mid_y),           # 0: Full Center
            (z2_x, 1968.0),             # 1: Bottom Half
            (z2_x, 2066.0)              # 2: Top Half
        ]

        # ROS Setup
        self.create_subscription(Poses2D, 'bot_pose', self.pose_cb, 10)
        self.create_subscription(Poses2D, 'crate_pose', self.crate_cb, 10)
        self.publisher = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
        self.timer = self.create_timer(0.03, self.control_cb)
        self.logging_timer = self.create_timer(1.0, self.log_bot_distances)

        # Service clients
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        self.assigned_positions = []

        self.get_logger().info("Multi-crate controller initialized for bots: " + ", ".join(map(str, self.bot_ids)))

        # Initialize arms
        rclpy.spin_once(self, timeout_sec=0.5)
        for _ in range(3):
            for bid in self.bot_ids:
                self.publish_wheel_velocities([0.0, 0.0, 0.0], bid, base=45.0, elbow=45.0)
                time.sleep(1)

    def trace_bot_loop(self, bot: BotState, state: str, extra: str = ""):
        if not self.enable_loop_trace_logs:
            return

        now = time.time()
        state_changed = (bot.loop_state != state)
        if state_changed:
            bot.loop_state = state
            bot.loop_state_enter_time = now
            bot.last_loop_state_log_time = 0.0

        if state_changed or (now - bot.last_loop_state_log_time) >= self.loop_trace_interval_s:
            elapsed = now - bot.loop_state_enter_time
            suffix = f" | {extra}" if extra else ""
            self.get_logger().info(f"[LOOP] Bot {bot.id} | {state} | for {elapsed:.1f}s{suffix}")
            bot.last_loop_state_log_time = now

    # ---------------- Callbacks ----------------
    def pose_cb(self, msg: Poses2D):
        for pose in msg.poses:
            if pose.id in self.bots:
                bot = self.bots[pose.id]
                now = self.get_clock().now()
                new_pose_np = np.array([pose.x, pose.y, pose.w])
                
                if bot.prev_pose_for_vel is not None:
                    dt = (now - bot.pose_timestamp).nanoseconds / 1e9
                    if dt > 0.001:
                        dx = new_pose_np[0] - bot.prev_pose_for_vel[0]
                        dy = new_pose_np[1] - bot.prev_pose_for_vel[1]
                        bot.velocity = math.hypot(dx, dy) / dt
                    else:
                        bot.velocity = 0.0
                
                bot.pose = new_pose_np
                bot.prev_pose_for_vel = new_pose_np
                bot.pose_timestamp = now
                bot.last_time = now

    def crate_cb(self, msg: Poses2D):
        self.crates = []
        for pose in msg.poses:
            self.crates.append({'id': pose.id, 'x': pose.x, 'y': pose.y, 'w': pose.w})
        if not self.drop_positions_assigned:
            self.assign_drop_positions()
            self.drop_positions_assigned = True

    def log_bot_distances(self):
        bots_with_pose = [b for b in self.bots.values() if b.pose is not None]
        for bot_a, bot_b in itertools.combinations(bots_with_pose, 2):
            try:
                dist = math.hypot(bot_a.pose[0] - bot_b.pose[0], bot_a.pose[1] - bot_b.pose[1])
                self.get_logger().info(f"Distance Bot {bot_a.id} <-> Bot {bot_b.id}: {dist:.2f}")
            except Exception as e:
                self.get_logger().error(f"Error calculating distance: {e}")

    # ---------------- Drop position assignment ----------------
    def get_zone_for_crate(self, crate_id):
        return crate_id % 3

    def assign_drop_positions(self):
        self.drop_positions = {}
        self.zone_last_positions = {}  # to track how many crates per zone

        for crate in self.crates:
            pos = self.generate_grid_position(crate['id'])
            self.drop_positions[crate['id']] = {'x': pos[0], 'y': pos[1] + 20}

        self.get_logger().info("Drop positions assigned in 2x2 grid pattern:")
        for cid, pos in self.drop_positions.items():
            self.get_logger().info(f"    Crate ID {cid}: ({pos['x']:.1f}, {pos['y']:.1f})")

    def generate_grid_position(self, crate_id):
        zone_id = self.get_zone_for_crate(crate_id)
        zone = self.drop_zones[zone_id]

        # Zone center
        center_x = (zone['x_min'] + zone['x_max']) / 2
        center_y = (zone['y_min'] + zone['y_max']) / 2

        # Distance between crates
        spacing_x = 160
        spacing_y = 120

        # Track how many crates already placed in this zone
        if zone_id not in self.zone_last_positions:
            self.zone_last_positions[zone_id] = 0
        count = self.zone_last_positions[zone_id]

        # Calculate relative offsets for 2x2 pattern
        offsets = [
            ( -spacing_x/2,  spacing_y/2),  # Crate 1 (top-left)
            (  spacing_x/2,  spacing_y/2),  # Crate 2 (top-right)
            ( -spacing_x/2, -spacing_y/2),  # Crate 3 (bottom-left)
            (  spacing_x/2, -spacing_y/2),  # Crate 4 (bottom-right)
        ]

        # If more than 4 crates, continue in next 2x2 block to the right
        block_index = count // 4
        crate_index = count % 4

        base_x_shift = block_index * (2 * spacing_x + 50)  # 50 gap between blocks
        offset_x, offset_y = offsets[crate_index]

        new_x = center_x + base_x_shift + offset_x
        new_y = center_y + offset_y

        # Update count for zone
        self.zone_last_positions[zone_id] += 1

        return (new_x, new_y)

    # ---------------- Greedy assignment ----------------
    def assign_crates_greedily(self):
        all_completed = set()
        for b in self.bots.values():
            all_completed |= b.completed_crates

        crates_in_use = set()
        for b in self.bots.values():
            if b.current_crate is not None:
                crates_in_use.add(b.current_crate['id'])
            if b.tracked_crate_id is not None and b.box_detached and b.current_crate is not None:
                crates_in_use.add(b.tracked_crate_id)

        unassigned_crates = [c for c in self.crates 
                            if c['id'] not in all_completed 
                            and c['id'] not in crates_in_use]

        if not unassigned_crates:
            return

        for bot in reversed(list(self.bots.values())):
            if bot.box_detached and bot.current_crate is not None:
                continue
            
            if bot.has_returned_home or bot.is_permanently_idle:
                continue
            
            if bot.current_crate is None and bot.pose is not None:
                nearest = min(unassigned_crates, key=lambda c: math.hypot(c['x'] - bot.pose[0], c['y'] - bot.pose[1]), default=None)
                if nearest:
                    bot.current_crate = nearest
                    bot.tracked_crate_id = nearest['id']
                    bot.goal_reached = False
                    bot.arm_placed = False
                    bot.box_attached = False
                    bot.arm_lifted = False
                    bot.move_after_attach = False
                    bot.box_detached = False
                    bot.current_path = []
                    bot.path_goal_cache = None
                    bot.last_path_calc_time = 0.0
                    bot.reset_pid()
                    zone_id = self.get_zone_for_crate(nearest['id'])
                    self.get_logger().info(f"Bot {bot.id} assigned to crate {nearest['id']} (Zone {zone_id})")
                    try:
                        unassigned_crates.remove(nearest)
                    except ValueError:
                        pass

    # ---------------- [FIXED FOLLOW BEHAVIOR COLLISION AVOIDANCE] ----------------
    # ---------------- [FIXED COLLISION LOGIC] ----------------
    def check_and_handle_collisions(self):
        bots_to_check = [b for b in self.bots.values() if b.pose is not None]
        bots_that_should_follow = set()

        # 1. Reset Crate Blocking Status
        for bot in bots_to_check:
            bot.blocked_by_crate = False

        # 2. Check Bot-vs-Bot Collisions (Standard Logic)
        for bot_a, bot_b in itertools.combinations(bots_to_check, 2):
            if not bot_a.is_moving and not bot_b.is_moving:
                continue
                
            dist = math.hypot(bot_a.pose[0] - bot_b.pose[0], bot_a.pose[1] - bot_b.pose[1])
            
            if dist < self.follow_distance_threshold:
                # Side-by-Side Exception (Same Zone Drops)
                if bot_a.current_crate is not None and bot_b.current_crate is not None:
                    id_a = bot_a.tracked_crate_id if bot_a.tracked_crate_id else bot_a.current_crate['id']
                    id_b = bot_b.tracked_crate_id if bot_b.tracked_crate_id else bot_b.current_crate['id']
                    if self.get_zone_for_crate(id_a) == self.get_zone_for_crate(id_b):
                        continue 

                # Leader/Follower Logic
                dist_a = math.hypot(bot_a.pose[0] - bot_a.goal[0], bot_a.pose[1] - bot_a.goal[1]) if bot_a.goal is not None else float('inf')
                dist_b = math.hypot(bot_b.pose[0] - bot_b.goal[0], bot_b.pose[1] - bot_b.goal[1]) if bot_b.goal is not None else float('inf')

                if dist_a < dist_b: leader, follower = bot_a, bot_b
                elif dist_b < dist_a: leader, follower = bot_b, bot_a
                else: leader, follower = (bot_a, bot_b) if bot_a.id > bot_b.id else (bot_b, bot_a)
                
                follower.is_following = True
                follower.follow_target_id = leader.id
                bots_that_should_follow.add(follower.id)

        # 3. Check Bot-vs-Crate Collisions
        for bot in bots_to_check:
            # Skip if already following a bot (don't double stop)
            if bot.is_following: continue
            
            # --- [CRITICAL PART] Identify my target ---
            my_target_id = -1
            if bot.tracked_crate_id is not None: 
                my_target_id = bot.tracked_crate_id
            elif bot.current_crate is not None: 
                my_target_id = bot.current_crate['id']

            for crate in self.crates:
                # --- [CRITICAL PART] The Exception ---
                # "Is this crate the one I am assigned to pick?"
                if crate['id'] == my_target_id: 
                    continue # YES -> Ignore safety, allow approach.
                
                # NO -> This is an obstacle crate. Check distance.
                dist_to_crate = math.hypot(bot.pose[0] - crate['x'], bot.pose[1] - crate['y'])

                if dist_to_crate < self.crate_avoidance_threshold:
                    self.get_logger().warn(f"Bot {bot.id} blocked by Obstacle Crate {crate['id']}! Stopping.")
                    bot.blocked_by_crate = True
                    break # Blocked, no need to check others
                
        # 4. Cleanup Follow Flags
        for bot in bots_to_check:
            if bot.id not in bots_that_should_follow and bot.is_following:
                bot.is_following = False
                bot.follow_target_id = None
    # ---------------- Main timer callback ----------------
    def control_cb(self):
        
        if len(self.crates) == 0:
            return
            
        # Reset slowdown scales
        for bot in self.bots.values():
            bot.slowdown_scale = 1.0

        # Refresh rasterized obstacle map each control cycle
        self.raster.clear_grid()
        for c in self.crates:
            self.raster.set_obstacle(c['x'], c['y'], 100.0)
        for b_state in self.bots.values():
            if b_state.pose is not None and b_state.is_permanently_idle:
                self.raster.set_obstacle(b_state.pose[0], b_state.pose[1], 150.0)

        # Assign crates
        self.assign_crates_greedily()

        # MODIFIED: Check for idle bots - ALL BOTS NOW USE STAGING LOGIC
        for bot in self.bots.values():
            if (bot.current_crate is None and 
                not bot.return_to_start and 
                not bot.move_after_attach and 
                not bot.going_to_staging_point and 
                not bot.has_returned_home and
                not bot.is_permanently_idle):

                # Check if bot needs staging based on Y position
                if bot.pose is not None and bot.pose[1] > self.staging_y_threshold:
                    bot.going_to_staging_point = True
                    
                    # Assign appropriate staging point based on bot ID and X position
                    if bot.id == 0:
                        # Bot 0: Check X position for staging point selection
                        if bot.pose[0] > 1219.0:
                            staging_goal = np.array([1700.0, 1219.0, 0.0])
                            self.get_logger().info(
                                f"Bot {bot.id} idle at y={bot.pose[1]:.1f}, x={bot.pose[0]:.1f} (> 1219), moving to staging point [1700, 1200, 0]")
                        else:
                            staging_goal = self.staging_point_0.copy()
                            self.get_logger().info(
                                f"Bot {bot.id} idle at y={bot.pose[1]:.1f}, x={bot.pose[0]:.1f} (<= 1219), moving to staging point [600, 1219, 0]")
                    
                    elif bot.id == 2:
                        # Bot 2: Check X position for staging point selection
                        if bot.pose[0] > 1219.0:
                            staging_goal = np.array([1700.0, 1500.0, 0.0])
                            self.get_logger().info(
                                f"Bot {bot.id} idle at y={bot.pose[1]:.1f}, x={bot.pose[0]:.1f} (> 1219), moving to staging point [1700, 1200, 0]")
                        else:
                            staging_goal = np.array([1700.0, 1500.0, 0.0])
                            self.get_logger().info(
                                f"Bot {bot.id} idle at y={bot.pose[1]:.1f}, x={bot.pose[0]:.1f} (<= 1219), moving to staging point [600, 1219, 0]")
                    
                    elif bot.id == 4:
                        # Bot 4: Check X position for staging point selection
                        if bot.pose[0] > 1219.0:
                            staging_goal = np.array([600.0, 1500.0, 0.0])
                            self.get_logger().info(
                                f"Bot {bot.id} idle at y={bot.pose[1]:.1f}, x={bot.pose[0]:.1f} (> 1219), moving to staging point [1700, 1200, 0]")
                        else:
                            staging_goal = np.array([600.0, 1500.0, 0.0])
                            self.get_logger().info(
                                f"Bot {bot.id} idle at y={bot.pose[1]:.1f}, x={bot.pose[0]:.1f} (<= 1219), moving to staging point [800, 1200, 0]")
                    else:
                        # Fallback for any other bot
                        staging_goal = np.array([600.0, 1219.0, 0.0])
                        self.get_logger().info(f"Bot {bot.id} idle, moving to default staging point")
                    
                    bot.goal = staging_goal
                
                else:
                    # Bot is below threshold, go home directly (skip staging)
                    bot.return_to_start = True
                    
                    # Set appropriate home point based on bot ID
                    if bot.id == 0:
                        bot.return_goal = self.idle_home_point_0
                        bot.goal = self.idle_home_point_0.copy()
                    elif bot.id == 2:
                        bot.return_goal = self.idle_home_point_2
                        bot.goal = self.idle_home_point_2.copy()
                    elif bot.id == 4:
                        bot.return_goal = self.idle_home_point_4
                        bot.goal = self.idle_home_point_4.copy()
                    else:
                        bot.return_goal = self.idle_home_point_0
                        bot.goal = self.idle_home_point_0.copy()
                    
                    if bot.pose is not None:
                        self.get_logger().info(f"Bot {bot.id} is idle at y={bot.pose[1]:.1f} (<= {self.staging_y_threshold}), returning home directly")
                    else:
                        self.get_logger().info(f"Bot {bot.id} is idle (no pose), returning home directly")

        # Check for collisions and set follow behavior
        self.check_and_handle_collisions()

        # Iterate through each bot's state machine
        for bot in self.bots.values():
            if bot.pose is None:
                self.trace_bot_loop(bot, "NO_POSE")
                continue

            # Handle follow behavior FIRST (highest priority)
            if bot.is_following and bot.follow_target_id is not None:
                self.trace_bot_loop(bot, "FOLLOW", f"leader={bot.follow_target_id}")
                self.get_logger().info(f"Bot {bot.id} executing FOLLOW behavior for Bot {bot.follow_target_id}")
                self.follow_bot(bot, bot.follow_target_id)
                continue
            # Handle Crate Avoidance (Second priority)
            if bot.blocked_by_crate:
                self.trace_bot_loop(bot, "BLOCKED_BY_CRATE")
                self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id)
                continue
            # Skip bots that are dropping
            if bot.box_detached and bot.current_crate is not None:
                self.trace_bot_loop(bot, "DROP_IN_PROGRESS", f"crate={bot.tracked_crate_id}")
                self.get_logger().debug(f"Bot {bot.id} is dropping crate {bot.tracked_crate_id}, skipping")
                continue
            
            # Handle staging point move for ALL BOTS
            if bot.going_to_staging_point:
                self.trace_bot_loop(bot, "GO_TO_STAGING")
                self.move_to_staging(bot)
                continue

            # Handle return home
            if bot.return_to_start:
                self.trace_bot_loop(bot, "RETURN_HOME")
                self.return_home(bot)
                continue

            # If no active crate, skip
            if bot.current_crate is None:
                self.trace_bot_loop(bot, "IDLE_NO_CRATE")
                continue
                
            # ... (inside the loop over bots) ...

            # ---------------------------------------------------------
            # NEW: SIMULTANEOUS ZONE LOGIC (Replaces Zone Locking)
            # ---------------------------------------------------------
            if bot.current_crate is not None:
                current_zone = self.get_zone_for_crate(bot.tracked_crate_id)
                
                # 1. Find all bots targeting this specific zone right now
                bots_in_zone = []
                for other_b in self.bots.values():
                    if other_b.current_crate is not None:
                        if self.get_zone_for_crate(other_b.tracked_crate_id) == current_zone:
                            bots_in_zone.append(other_b)
                
                # Sort by ID to ensure consistent assignment (Bot 0 always goes to Subpart 1, Bot 2 to Subpart 2)
                bots_in_zone.sort(key=lambda b: b.id)

                # 2. Check overlap and assign subparts
                if len(bots_in_zone) > 1:
                    # collision imminent: split them up!
                    # The first bot in the list gets Subpart 1, second gets Subpart 2
                    try:
                        my_index = bots_in_zone.index(bot)
                        
                        # Select Subpart (Index 1 or 2 based on list position)
                        subpart_idx = my_index + 1  
                        
                        # Safety: If 3 bots somehow target same zone, fallback to index 1
                        if subpart_idx > 2: subpart_idx = 1 

                        target_coords = self.zone_subparts[current_zone][subpart_idx]
                        
                        # UPDATE THE GOAL DIRECTLY
                        # We override the drop_position so move_to_final_goal picks it up
                        self.drop_positions[bot.tracked_crate_id]['x'] = target_coords[0]
                        self.drop_positions[bot.tracked_crate_id]['y'] = target_coords[1]
                        
                        self.get_logger().info(f"Bot {bot.id}: Simultaneous Zone {current_zone} access! Redirecting to Subpart {subpart_idx}")
                        
                    except ValueError:
                        pass
                else:
                    # I am alone in this zone: Use the standard/full goal (Subpart 0)
                    # Note: You might prefer to keep the grid position from generate_grid_position here.
                    # If you strictly want 'Subpart 0' (Center) when alone, uncomment below:
                    # target_coords = self.zone_subparts[current_zone][0]
                    # self.drop_positions[bot.tracked_crate_id]['x'] = target_coords[0]
                    # self.drop_positions[bot.tracked_crate_id]['y'] = target_coords[1]
                    pass

            # ---------------------------------------------------------
            # OLD LOCKING LOGIC REMOVED
            # (Deleted the 'if zone_locks.get(zone_id)...' block)
            # ---------------------------------------------------------

            # Run state machine
            if not bot.goal_reached:
                 self.trace_bot_loop(bot, "MOVE_NEAR_CRATE", f"crate={bot.tracked_crate_id}")
                 self.move_near_crate(bot)
            # ... rest of state machine ...
            # Run state machine
            if not bot.goal_reached:
                self.trace_bot_loop(bot, "MOVE_NEAR_CRATE", f"crate={bot.tracked_crate_id}")
                self.move_near_crate(bot)
            elif not bot.arm_placed:
                self.trace_bot_loop(bot, "PLACE_ARM", f"crate={bot.tracked_crate_id}")
                self.place_arm_on_crate(bot)
            elif not bot.box_attached:
                self.trace_bot_loop(bot, "ATTACH_CRATE", f"crate={bot.tracked_crate_id}")
                self.attach_crate(bot)
            elif not bot.arm_lifted:
                self.trace_bot_loop(bot, "LIFT_ARM", f"crate={bot.tracked_crate_id}")
                self.lift_arm_with_crate(bot)
            elif bot.move_after_attach:
                self.trace_bot_loop(bot, "MOVE_TO_DROP", f"crate={bot.tracked_crate_id}")
                self.move_to_final_goal(bot)
        # -----------------------------------------------------------
        # NEW: GATHER DATA FOR VISUALIZATION
        # -----------------------------------------------------------
        # Structure: { zone_id: { subpart_id: bot_id } }
        viz_data = {0: {}, 1: {}, 2: {}}

        for bot in self.bots.values():
            if bot.current_crate is not None:
                # 1. Determine which zone
                zid = self.get_zone_for_crate(bot.tracked_crate_id)
                
                # 2. Determine which subpart (Re-running the logic briefly)
                bots_in_zone = [b for b in self.bots.values() 
                                if b.current_crate and self.get_zone_for_crate(b.tracked_crate_id) == zid]
                bots_in_zone.sort(key=lambda b: b.id)
                
                subpart = 0 # Default (Center)
                if len(bots_in_zone) > 1:
                    try:
                        subpart = bots_in_zone.index(bot) + 1
                    except: pass
                
                # 3. Add to data
                viz_data[zid][subpart] = bot.id

        # Publish
        msg = String()
        msg.data = json.dumps(viz_data)
        self.viz_publisher.publish(msg)

    # ---------------- Follow Behavior Function ----------------
    def follow_bot(self, follower: BotState, leader_id: int):
        """
        Makes the follower bot follow the leader bot at a safe distance.
        """
        if leader_id not in self.bots:
            self.get_logger().warn(f"Bot {follower.id}: Leader {leader_id} not found!")
            return
        
        leader = self.bots[leader_id]
        if leader.pose is None or follower.pose is None:
            self.get_logger().warn(f"Bot {follower.id}: Missing pose data (leader or follower)")
            return
        
        follower.is_moving = True
        
        # Calculate direction from follower to leader
        dx = leader.pose[0] - follower.pose[0]
        dy = leader.pose[1] - follower.pose[1]
        dist_to_leader = math.hypot(dx, dy)
        
        self.get_logger().info(f"Bot {follower.id} following Bot {leader.id}: distance = {dist_to_leader:.1f}")
        
        # If already at good distance (within tolerance), just stop
        tolerance = 20.0
        if abs(dist_to_leader - follower.follow_offset_distance) < tolerance:
            self.get_logger().info(f"Bot {follower.id}: At good follow distance, stopping")
            self.publish_wheel_velocities([0.0, 0.0, 0.0], follower.id)
            follower.is_moving = False
            return
        
        # Calculate the desired follow position behind the leader
        # We want to be at follow_offset_distance away from the leader
        if dist_to_leader < 1.0:
            # Too close, can't calculate direction
            self.get_logger().info(f"Bot {follower.id}: Too close to leader, stopping")
            self.publish_wheel_velocities([0.0, 0.0, 0.0], follower.id)
            follower.is_moving = False
            return
        
        # Unit direction vector from follower to leader
        dir_x = dx / dist_to_leader
        dir_y = dy / dist_to_leader
        
        # Target position: stay at follow_offset_distance behind leader
        # Position ourselves along the line between us and the leader
        target_x = leader.pose[0] - dir_x * follower.follow_offset_distance
        target_y = leader.pose[1] - dir_y * follower.follow_offset_distance
        
        self.get_logger().info(
            f"Bot {follower.id}: Current=({follower.pose[0]:.1f}, {follower.pose[1]:.1f}), "
            f"Leader=({leader.pose[0]:.1f}, {leader.pose[1]:.1f}), "
            f"Target=({target_x:.1f}, {target_y:.1f})"
        )
        
        # Create temporary follow goal
        follow_goal = np.array([target_x, target_y, 0.0])
        
        # Use the standard movement function
        self.move_to_point(follower, follow_goal, base=45.0, elbow=45.0)

    # ---------------- Per-bot actions ----------------
    def move_near_crate(self, bot: BotState):
        bot.is_moving = True
        crate_pose = bot.current_crate
        if crate_pose is None:
            return

        # --- NEW: Dynamic Alignment Logic ---
        # 1. Get crate orientation
        crate_angle_deg = crate_pose['w']
        crate_angle_rad = math.radians(crate_angle_deg)

        # 2. Rotate the approach vector (0, -130) by the crate's angle
        # This keeps the robot in front of the "Face" regardless of rotation
        dist_from_center = 130.0
        
        offset_x = -(-dist_from_center) * math.sin(crate_angle_rad)
        offset_y = +(-dist_from_center) * math.cos(crate_angle_rad)

        # 3. Set Goal Position & Angle
        goal_x = crate_pose['x'] + offset_x
        goal_y = crate_pose['y'] + offset_y
        goal_theta_deg = crate_angle_deg 

        bot.goal = np.array([goal_x, goal_y, goal_theta_deg])
        # ------------------------------------

        now = self.get_clock().now()
        dt = (now - bot.last_time).nanoseconds / 1e9 if bot.last_time is not None else 0.03
        if dt <= 0:
            return

        x, y, theta_deg = bot.pose
        theta_rad = math.radians(theta_deg)
        goal_theta_rad = math.radians(goal_theta_deg)

        error_x = goal_x - x
        error_y = goal_y - y
        error_theta = (goal_theta_rad - theta_rad + math.pi) % (2 * math.pi) - math.pi

        vx_world = bot.pid_x.compute(error_x, dt)
        vy_world = bot.pid_y.compute(error_y, dt)
        omega = bot.pid_theta.compute(error_theta, dt)
        
        vx_world *= bot.slowdown_scale
        vy_world *= bot.slowdown_scale
        omega *= bot.slowdown_scale

        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        vx_robot = vx_world * cos_t + vy_world * sin_t
        vy_robot = -vx_world * sin_t + vy_world * cos_t

        alphas = [math.radians(30), math.radians(150), math.radians(270)]
        M = np.array([
            [math.cos(a + math.pi/2) for a in alphas],
            [math.sin(a + math.pi/2) for a in alphas],
            [1.0, 1.0, 1.0]
        ])
        v = np.array([vx_robot, vy_robot, omega])
        wheel_vel = np.linalg.pinv(M) @ v

        self.publish_wheel_velocities(wheel_vel.tolist(), bot.id)
        dist = math.hypot(error_x, error_y)

        # Check success (Distance AND Angle)
        if dist < self.xy_tolerance and abs(error_theta) < math.radians(self.theta_tolerance_deg):
            self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id)
            bot.is_moving = False
            bot.goal_reached = True
            self.get_logger().info(f"Bot {bot.id} aligned to crate {bot.tracked_crate_id} at {goal_theta_deg:.1f} deg")
    def place_arm_on_crate(self, bot: BotState):
        base_angle = 90.0
        elbow_angle = 90.0
        self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id, base=base_angle, elbow=elbow_angle)
        bot.arm_placed = True
        time.sleep(3)
        self.get_logger().info(f"Bot {bot.id} arm placed on crate {bot.tracked_crate_id}.")

    def get_model1_name(self, bot_id: int):
        if bot_id == 0:
            return "hb_crystal"
        elif bot_id == 4:
            return "hb_glacio"
        elif bot_id == 2:
            return "hb_frostbite"
        else:
            return "hb_crystal"
        
    def attach_crate(self, bot: BotState):
        if not self.attach_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("/attach_link service unavailable.")
            return

        color_mod = self.get_zone_for_crate(bot.tracked_crate_id)
        
        if color_mod == 0:
            model2_name = f"crate_red_{bot.tracked_crate_id}"
        elif color_mod == 1:
            model2_name = f"crate_green_{bot.tracked_crate_id}"
        else:
            model2_name = f"crate_blue_{bot.tracked_crate_id}"

        link2_name = f"box_link_{bot.tracked_crate_id}"
        req = AttachLink.Request()
        model1_name = self.get_model1_name(bot.id)

        self.get_logger().info(
            f"[ATTACH] Bot {bot.id} → model1='{model1_name}', model2='{model2_name}'"
        )

        req.data = f"""{{
            "model1_name": "{model1_name}",
            "link1_name": "arm_link_2",
            "model2_name": "{model2_name}",
            "link2_name": "{link2_name}"
        }}"""
        future = self.attach_client.call_async(req)
        future.add_done_callback(lambda fut, b=bot: self.attach_done_callback(fut, b))
        bot.box_attached = True
        self.get_logger().info(f"Bot {bot.id} requested attach for crate {bot.tracked_crate_id}.")

    def lift_arm_with_crate(self, bot: BotState):
        base_angle = 45.0
        elbow_angle = 45.0
        self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id, base=base_angle, elbow=elbow_angle)
        bot.arm_lifted = True
        bot.move_after_attach = True
        self.get_logger().info(f"Bot {bot.id} arm lifted with crate attached!")

    def move_to_final_goal(self, bot: BotState):
        drop_pos = self.drop_positions.get(bot.tracked_crate_id, None)
        if drop_pos is None:
            goal = bot.final_goal
        else:
            goal = np.array([drop_pos['x'], (drop_pos['y'] - 190), 0.0])

        bot.goal = goal.copy()
        if self.move_to_point(bot, goal, base=45.0, elbow=45.0):
            bot.move_after_attach = False
            self.get_logger().info(f"Bot {bot.id} reached final goal. Dropping crate {bot.tracked_crate_id}...")
            self.drop_crate(bot)

    # MODIFIED: Move to staging for ALL BOTS
    def move_to_staging(self, bot: BotState):
        """
        Moves ALL bots to their intermediate staging point.
        """
        if self.move_to_point(bot, bot.goal, base=45.0, elbow=45.0):
            self.get_logger().info(f"Bot {bot.id} reached staging point. Now returning home.")
            bot.going_to_staging_point = False
            bot.return_to_start = True
            
            # Set appropriate home point based on bot ID
            if bot.id == 0:
                bot.return_goal = self.idle_home_point_0
                bot.goal = self.idle_home_point_0.copy()
            elif bot.id == 2:
                bot.return_goal = self.idle_home_point_2
                bot.goal = self.idle_home_point_2.copy()
            elif bot.id == 4:
                bot.return_goal = self.idle_home_point_4
                bot.goal = self.idle_home_point_4.copy()
            else:
                bot.return_goal = self.idle_home_point_0
                bot.goal = self.idle_home_point_0.copy()
            
            bot.reset_pid()

    def return_home(self, bot: BotState):
        if self.move_to_point(bot, bot.return_goal, base=45.0, elbow=45.0):
            bot.return_to_start = False
            bot.has_returned_home = True
            bot.is_permanently_idle = True
            bot.goal = bot.return_goal.copy()
            self.get_logger().info(f"Bot {bot.id} returned to start position and is now permanently idle.")
            self.reset_bot_state(bot)

    # ---------------- Helper Movement Function ----------------
    def move_to_point(self, bot: BotState, goal, base=0.0, elbow=0.0):
        bot.is_moving = True
        now = self.get_clock().now()
        dt = (now - bot.last_time).nanoseconds / 1e9 if bot.last_time is not None else 0.03
        if dt <= 0:
            return False

        x, y, theta_deg = bot.pose
        theta_rad = math.radians(theta_deg)
        goal_x, goal_y, goal_theta_deg = goal
        goal_theta_rad = math.radians(goal_theta_deg)

        # A* replanning trigger
        target_coords = (goal_x, goal_y)
        needs_recalc = False
        if not bot.current_path:
            needs_recalc = True
        elif bot.path_goal_cache is None:
            needs_recalc = True
        elif math.hypot(goal_x - bot.path_goal_cache[0], goal_y - bot.path_goal_cache[1]) > 50.0:
            needs_recalc = True
        elif time.time() - bot.last_path_calc_time > 1.0:
            needs_recalc = True

        if needs_recalc:
            bot.current_path, _ = a_star_search(self.raster, (x, y), target_coords)
            bot.path_goal_cache = target_coords
            bot.last_path_calc_time = time.time()

        # Path follower with lookahead
        if bot.current_path and len(bot.current_path) > 0:
            lookahead_dist = 120.0
            waypoint_tol = 40.0
            target_pt = bot.current_path[-1]

            while len(bot.current_path) > 1:
                dist_to_first = math.hypot(bot.current_path[0][0] - x, bot.current_path[0][1] - y)
                if dist_to_first < waypoint_tol:
                    bot.current_path.pop(0)
                else:
                    break

            for pt in bot.current_path:
                d = math.hypot(pt[0] - x, pt[1] - y)
                if d > lookahead_dist:
                    target_pt = pt
                    break

            goal_x = target_pt[0]
            goal_y = target_pt[1]
            if target_pt != bot.current_path[-1]:
                bot.slowdown_scale = 1.0

        error_x = goal_x - x
        error_y = goal_y - y
        error_theta = (goal_theta_rad - theta_rad + math.pi) % (2 * math.pi) - math.pi

        vx_world = bot.pid_x.compute(error_x, dt)
        vy_world = bot.pid_y.compute(error_y, dt)
        omega = bot.pid_theta.compute(error_theta, dt)

        vx_world *= bot.slowdown_scale
        vy_world *= bot.slowdown_scale
        omega *= bot.slowdown_scale

        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        vx_robot = vx_world * cos_t + vy_world * sin_t
        vy_robot = -vx_world * sin_t + vy_world * cos_t

        alphas = [math.radians(30), math.radians(150), math.radians(270)]
        M = np.array([
            [math.cos(a + math.pi/2) for a in alphas],
            [math.sin(a + math.pi/2) for a in alphas],
            [1.0, 1.0, 1.0]
        ])
        v = np.array([vx_robot, vy_robot, omega])
        wheel_vel = np.linalg.pinv(M) @ v

        self.publish_wheel_velocities(wheel_vel.tolist(), bot.id, base=base, elbow=elbow)

        dist = math.hypot(error_x, error_y)
        if dist < (self.xy_tolerance + 3) and abs(error_theta) < math.radians(self.theta_tolerance_center_deg):
            bot.is_moving = False
            self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id, base=base, elbow=elbow)
            return True
        return False

    # ---------------- Drop Sequence ----------------
    def drop_crate(self, bot: BotState):
        self.get_logger().info(f"Bot {bot.id} DROPPING crate {bot.tracked_crate_id} - LOCKING from assignment")
        
        base_angle = 85.0
        elbow_angle = 85.0
        self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id, base=base_angle, elbow=elbow_angle)
        time.sleep(3)

        if not self.detach_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("/detach_link service unavailable.")
            return

        color_mod = self.get_zone_for_crate(bot.tracked_crate_id)

        if color_mod == 0:
            model2_name = f"crate_red_{bot.tracked_crate_id}"
        elif color_mod == 1:
            model2_name = f"crate_green_{bot.tracked_crate_id}"
        else:
            model2_name = f"crate_blue_{bot.tracked_crate_id}"

        link2_name = f"box_link_{bot.tracked_crate_id}"
        req = DetachLink.Request()
        model1_name = self.get_model1_name(bot.id)

        req.data = f"""{{
            "model1_name": "{model1_name}",
            "link1_name": "arm_link_2",
            "model2_name": "{model2_name}",
            "link2_name": "{link2_name}"
        }}"""

        future = self.detach_client.call_async(req)
        future.add_done_callback(lambda fut, b=bot: self.detach_done_callback(fut, b))
        bot.box_detached = True

        time.sleep(0.8)
        lift_base, lift_elbow = 45.0, 45.0
        self.publish_wheel_velocities([0.0, 0.0, 0.0], bot.id, base=lift_base, elbow=lift_elbow)

        drop_pos = self.drop_positions.get(bot.tracked_crate_id, None)
        if drop_pos:
            self.get_logger().info(
                f"Bot {bot.id}: Crate ID {bot.tracked_crate_id} placed at "
                f"({drop_pos['x']:.1f}, {drop_pos['y']:.1f}) successfully."
            )
        else:
            self.get_logger().info(f"Bot {bot.id}: Crate ID {bot.tracked_crate_id} placed successfully.")

    # ---------------- Service callbacks ----------------
    def attach_done_callback(self, future, bot: BotState):
        try:
            res = future.result()
            if hasattr(res, 'success') and res.success:
                self.get_logger().info(f"Bot {bot.id}: Box attached successfully.")
            else:
                self.get_logger().warn(f"Bot {bot.id}: Attach failed.")
        except Exception as e:
            self.get_logger().error(f"Bot {bot.id}: Attach failed: {e}")

    def detach_done_callback(self, future, bot: BotState):
        try:
            res = future.result()
            if hasattr(res, 'success') and res.success:
                self.get_logger().info(f"Bot {bot.id}: Crate detached successfully.")
                """""
                # Unlock the zone
                try:
                    zone_id = self.get_zone_for_crate(bot.tracked_crate_id)
                    if self.zone_locks.get(zone_id) == bot.id:
                        self.zone_locks[zone_id] = None
                        self.get_logger().info(f"Bot {bot.id} has UNLOCKED zone {zone_id}.")
                    elif self.zone_locks.get(zone_id) is not None:
                        self.get_logger().warn(f"Bot {bot.id} finished in zone {zone_id}, but lock was held by {self.zone_locks.get(zone_id)}!")
                except Exception as e:
                    self.get_logger().error(f"Error during zone unlock: {e}")
"""
                # Mark as completed BEFORE clearing current_crate
                if bot.current_crate is not None:
                    bot.completed_crates.add(bot.current_crate['id'])
                    self.get_logger().info(f"Bot {bot.id}: Marked crate {bot.current_crate['id']} as COMPLETED")
                
                # Clear current_crate
                bot.current_crate = None
                bot.tracked_crate_id = None

                # Reset flags
                bot.goal_reached = False
                bot.arm_placed = False
                bot.box_attached = False
                bot.arm_lifted = False
                bot.has_returned_home = False
                bot.move_after_attach = False
                bot.box_detached = False
                bot.current_path = []
                bot.path_goal_cache = None
                bot.last_path_calc_time = 0.0
                bot.reset_pid()
                
                # Priority cooldown
                bot.priority_cooldown_end_time = self.get_clock().now() + Duration(seconds=1)
                self.get_logger().info(f"Bot {bot.id} has priority cooldown")

            else:
                self.get_logger().warn(f"Bot {bot.id}: Detach failed.")
        except Exception as e:
            self.get_logger().error(f"Bot {bot.id}: Detach failed: {e}")

    # ---------------- Utilities ----------------
    def reset_bot_state(self, bot: BotState):
        """
        General reset, used when returning home.
        Does NOT reset has_returned_home or is_permanently_idle flags.
        """
        bot.goal_reached = False
        bot.arm_placed = False
        bot.box_attached = False
        bot.arm_lifted = False
        bot.goal = None
        bot.move_after_attach = False
        bot.box_detached = False
        bot.going_to_staging_point = False
        bot.is_moving = False
        bot.is_following = False
        bot.follow_target_id = None
        bot.current_path = []
        bot.path_goal_cache = None
        bot.last_path_calc_time = 0.0
        bot.reset_pid()

    def publish_wheel_velocities(self, wheel_vel, bot_id, base=45.0, elbow=45.0):
        msg = BotCmdArray()
        cmd = BotCmd()
        cmd.id = bot_id
        if len(wheel_vel) >= 3:
            cmd.m1, cmd.m2, cmd.m3 = wheel_vel[0], wheel_vel[1], wheel_vel[2]
        else:
            cmd.m1 = wheel_vel[0] if len(wheel_vel) > 0 else 0.0
            cmd.m2 = wheel_vel[1] if len(wheel_vel) > 1 else 0.0
            cmd.m3 = 0.0
        cmd.base = base
        cmd.elbow = elbow
        msg.cmds.append(cmd)
        self.publisher.publish(msg)


# ---------------------- Main -------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = HolonomicMoveToCratesMulti()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down multi-bot node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
