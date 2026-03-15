#!/usr/bin/env python3

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from hb_interfaces.msg import Pose2D, Poses2D
from std_msgs.msg import String
import json
import yaml
import os

# ---------------- CONFIGURATION ----------------
SIMULATION_MODE = True
# -----------------------------------------------

# Arena layout: IDs and their real-world coordinates in meters (X, Y)
ARENA_LAYOUT = {
    1: [0.0, 0.0],      # Top-left corner
    3: [2.4384, 0.0],   # Top-right corner
    7: [2.4384, 2.4384],# Bottom-right corner
    5: [0.0, 2.4384]    # Bottom-left corner
}

class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')
        
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ---------- PARAMETERS ----------
        self.bots_marker_length = 0.05
        self.aruco_dict_name = 'DICT_4X4_50'
        self.arena_min = 0.0
        self.arena_max = 2438.4
        
        # ---------- INITIALIZATION BASED ON MODE ----------
        if SIMULATION_MODE:
            self.get_logger().info("STARTING PERCEPTION IN SIMULATION MODE")
            # Simulation subscribes to CameraInfo for calibration
            self.image_topic = "/camera/image_raw"
            self.camera_info_sub = self.create_subscription(
                CameraInfo, "/camera/camera_info", self.camera_info_callback, 1
            )
        else:
            self.get_logger().info("STARTING PERCEPTION IN REAL HARDWARE MODE")
            self.image_topic = "/image_raw"
            # Real hardware loads calibration from YAML
            self.load_real_calibration()

        # ---------- COMMON TOPICS ----------
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.crate_poses_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.bot_poses_pub = self.create_publisher(Poses2D, '/bot_pose', 10)
        self.camera_debug_pub = self.create_publisher(String, '/camera_debug', 10)
        
        # Subscribe to the controller's debug output to draw the HUD
        self.debug_sub = self.create_subscription(String, '/perception_debug', self.debug_callback, 10)
        self.latest_debug_data = {} 

        # ---------- ARUCO SETUP ----------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Matrices for Homography
        self.pixel_matrix = []
        self.world_matrix = []
        self.H_matrix = None
        
        self.get_logger().info('PoseDetector initialized')

    def load_real_calibration(self):
        """ Loads calibration data from YAML file for Real Life mode """
        calib_file = "/home/harry/calibration_data2/ost.yaml"
        try:
            with open(calib_file, 'r') as f:
                calib_data = yaml.safe_load(f)
            self.camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
            self.dist_coeffs = np.array(calib_data['distortion_coefficients']['data'])
            self.get_logger().info("Loaded REAL camera calibration from YAML")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration YAML: {e}")

    def camera_info_callback(self, msg):
        """ Callback for Simulation CameraInfo """
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Simulation Camera parameters loaded.')
            self.destroy_subscription(self.camera_info_sub)

    def debug_callback(self, msg):
        """ Receives logic state from controller for HUD drawing """
        try:
            self.latest_debug_data = json.loads(msg.data)
        except: pass

    def pixel_to_world(self, pixel_x, pixel_y):
        """ transform pixel (u,v) to world (x_mm, y_mm) """
        if self.H_matrix is None:
            return None, None
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point, self.H_matrix)
        
        # Homography was calculated with Meters, convert to mm
        world_x_mm = world_point[0][0][0] * 1000.0
        world_y_mm = world_point[0][0][1] * 1000.0
        return world_x_mm, world_y_mm

    def world_to_pixel(self, wx, wy):
        """ transform world (x_mm, y_mm) to pixel (u,v) """
        if self.H_matrix is None: return None, None
        try:
            H_inv = np.linalg.inv(self.H_matrix)
            # Convert mm back to meters for the transform
            world_pt = np.array([[[wx/1000.0, wy/1000.0]]], dtype=np.float32)
            pixel_pt = cv2.perspectiveTransform(world_pt, H_inv)
            return int(pixel_pt[0][0][0]), int(pixel_pt[0][0][1])
        except:
            return None, None

    # ---------------- DRAW HUD ----------------
    def draw_hud(self, img, ids, corners, bot_poses, crate_poses):
        
        # 1. DRAW ZONES (Static Definitions)
        # Drop Zones
        self.draw_zone_box(img, 0, 1020, 1400, 1075, 1355, zone_type="drop")
        self.draw_zone_box(img, 1, 675, 920, 1920, 2115, zone_type="drop")
        self.draw_zone_box(img, 2, 1470, 1752, 1920, 2115, zone_type="drop")

        # Pickup Zones
        pickup_zones = [
            {'id': 0, 'x_min': 180, 'x_max': 390, 'y_min': 560, 'y_max': 1111},
            {'id': 1, 'x_min': 2060, 'x_max': 2266, 'y_min': 560, 'y_max': 1111},
            {'id': 2, 'x_min': 180, 'x_max': 390, 'y_min': 1350, 'y_max': 1900},
            {'id': 3, 'x_min': 2060, 'x_max': 2266, 'y_min': 1350, 'y_max': 1900}
        ]
        for pz in pickup_zones:
            self.draw_zone_box(img, pz['id'], pz['x_min'], pz['x_max'], 
                               pz['y_min'], pz['y_max'], zone_type="pickup")

        # 2. PREPARE DATA
        controller_data = self.latest_debug_data.get('bots', {})
        drop_data = self.latest_debug_data.get('drop_positions', {})
        bot_pixel_centers = {} 

        if ids is not None:
            for i, aruco_id in enumerate(ids.flatten()):
                sid = str(aruco_id)
                iid = int(aruco_id)
                
                # Get Pixel Center
                c = corners[i][0]
                cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))

                # --- BOTS ---
                if iid in bot_poses: 
                    bot_pixel_centers[iid] = (cx, cy) 
                    pose = bot_poses[iid]
                    
                    # Coordinate Text
                    coord_text = f"B{iid} X:{int(pose['x'])} Y:{int(pose['y'])} W:{int(pose['yaw'])}"
                    cv2.putText(img, coord_text, (cx - 55, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    if sid in controller_data:
                        info = controller_data[sid]
                        goal_arr = info.get('goal', [])
                        gx_active = info.get('goal_x')
                        gy_active = info.get('goal_y')
                        goal_text = ""

                        # Draw Goal Line
                        if len(goal_arr) >= 2 and gx_active is not None and gy_active is not None:
                            gx_global, gy_global = goal_arr[0], goal_arr[1]
                            dx = gx_active - pose['x']
                            dy = gy_active - pose['y']
                            
                            goal_text = f" D:[{int(dx)},{int(dy)}]"
                            
                            # Detour Detection
                            is_detouring = (abs(gx_active - gx_global) > 1.0) or (abs(gy_active - gy_global) > 1.0)
                            
                            p_global = self.world_to_pixel(gx_global, gy_global)
                            p_active = self.world_to_pixel(gx_active, gy_active)
                            
                            if p_global[0] is not None:
                                if is_detouring and p_active[0] is not None:
                                    # Detour: Green line to global, Orange to active
                                    cv2.line(img, (cx, cy), p_global, (0, 100, 0), 1)
                                    cv2.line(img, (cx, cy), p_active, (0, 165, 255), 2)
                                    cv2.putText(img, "DETOUR", (p_active[0], p_active[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                                else:
                                    # Standard: Green line to goal
                                    cv2.line(img, (cx, cy), p_global, (0, 255, 0), 1)
                                    cv2.putText(img, "G", (p_global[0], p_global[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        # Status Text
                        status_label = f"ID:{aruco_id} {info.get('status', 'IDLE')}{goal_text}"
                        cv2.putText(img, status_label, (cx + 25, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # --- CRATES ---
                elif iid in crate_poses:
                    pose = crate_poses[iid]
                    cyan = (255, 255, 0)
                    cv2.rectangle(img, (cx-20, cy-20), (cx+20, cy+20), cyan, 2)
                    cv2.putText(img, f"CRATE {iid}", (cx-35, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2)
                    
                    coord_text = f"X:{int(pose['x'])} Y:{int(pose['y'])}"
                    cv2.putText(img, coord_text, (cx - 40, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cyan, 1)

                    if sid in drop_data:
                        tgt = drop_data[sid]
                        tx, ty = self.world_to_pixel(tgt['x'], tgt['y'])
                        if tx is not None:
                            cv2.line(img, (cx, cy), (tx, ty), cyan, 1)
                            cv2.putText(img, "DROP", (tx+15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2)

        # 3. DRAW DISTANCE LINES (Bot-to-Bot)
        detected_bots = list(bot_pixel_centers.keys())
        for i in range(len(detected_bots)):
            for j in range(i + 1, len(detected_bots)):
                id1 = detected_bots[i]
                id2 = detected_bots[j]
                pt1 = bot_pixel_centers[id1]
                pt2 = bot_pixel_centers[id2]
                
                # Real world distance
                p1_real = bot_poses[id1] 
                p2_real = bot_poses[id2]
                dist_mm = math.sqrt( (p1_real['x'] - p2_real['x'])**2 + (p1_real['y'] - p2_real['y'])**2 )
                dist_m = dist_mm / 1000.0

                cv2.line(img, pt1, pt2, (255, 0, 255), 2)
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)
                cv2.putText(img, f"{dist_m:.2f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    def draw_zone_box(self, img, zone_id, x_min, x_max, y_min, y_max, zone_type="drop"):
        tl = self.world_to_pixel(x_min, y_min)
        tr = self.world_to_pixel(x_max, y_min)
        br = self.world_to_pixel(x_max, y_max)
        bl = self.world_to_pixel(x_min, y_max)

        if tl[0] is None: return

        # Dynamic Lock Lookup
        locked_by = None
        if zone_type == "drop":
            locks = self.latest_debug_data.get('drop_zones', {}).get('locks', {})
            locked_by = locks.get(str(zone_id))
            label_prefix = "DROP"
        else: 
            locks = self.latest_debug_data.get('drop_zones', {}).get('pickup_zone_locks', {})
            locked_by = locks.get(str(zone_id))
            label_prefix = "PICK"
        
        # Color Logic
        if locked_by is not None: 
            color = (0, 0, 255) # Red
            text = f"{label_prefix} {zone_id}: LOCKED ({locked_by})"
            thickness = 2
        else:
            color = (0, 255, 0) # Green
            text = f"{label_prefix} {zone_id}: FREE"
            thickness = 1

        # Draw
        pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, thickness)
        
        # Label
        cx = int((tl[0] + br[0]) / 2)
        cy = int((tl[1] + br[1]) / 2)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, text, (cx - (text_w//2), cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ---------------- IMAGE CALLBACK ----------------
    def image_callback(self, msg):
        try:
            if self.camera_matrix is None or self.dist_coeffs is None:
                if SIMULATION_MODE: self.get_logger().warn_once("Waiting for camera parameters...")
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            undistorted_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            refined_corners = corners
            if ids is not None:
                # Subpixel refinement
                all_corners_flat = np.concatenate(corners, axis=1).reshape(-1, 2).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_flat = cv2.cornerSubPix(gray, all_corners_flat, (5, 5), (-1, -1), criteria)
                
                start = 0
                refined_corners = []
                for mc in corners:
                    num = len(mc[0])
                    end = start + num
                    refined_corners.append(refined_flat[start:end].reshape(1, num, 2))
                    start = end
                
                cv2.aruco.drawDetectedMarkers(undistorted_image, refined_corners, ids)

                # --- HOMOGRAPHY CALCULATION ---
                self.pixel_matrix = []
                self.world_matrix = []
                for i, aruco_id in enumerate(ids.flatten()):
                    if aruco_id in ARENA_LAYOUT:
                        c = refined_corners[i][0]
                        # Top-left, Top-Right, Bottom-Right, Bottom-Left logic
                        if aruco_id == 1:   px, py, wx, wy = c[0][0], c[0][1], 0.0, 0.0
                        elif aruco_id == 3: px, py, wx, wy = c[1][0], c[1][1], 2.4384, 0.0
                        elif aruco_id == 7: px, py, wx, wy = c[2][0], c[2][1], 2.4384, 2.4384
                        elif aruco_id == 5: px, py, wx, wy = c[3][0], c[3][1], 0.0, 2.4384
                        self.pixel_matrix.append([px, py])
                        self.world_matrix.append([wx, wy])

                if len(self.pixel_matrix) >= 4:
                    pp = np.array(self.pixel_matrix, dtype=np.float32)
                    wp = np.array(self.world_matrix, dtype=np.float32)
                    self.H_matrix, _ = cv2.findHomography(pp, wp, cv2.RANSAC, 5.0)

                # --- POSE ESTIMATION ---
                if self.H_matrix is not None:
                    bot_poses = {}
                    crate_poses = {}
                    
                    for i, aruco_id in enumerate(ids.flatten()):
                        if aruco_id not in ARENA_LAYOUT:
                            marker_corners = refined_corners[i][0]
                            cx, cy = np.mean(marker_corners[:, 0]), np.mean(marker_corners[:, 1])
                            x, y = self.pixel_to_world(cx, cy)

                            # --- HEIGHT CORRECTION LOGIC (Unified) ---
                            center_mm = 1219.2
                            
                            if SIMULATION_MODE:
                                cam_h = 2438.4
                                if aruco_id == 9: # Simulation Crate
                                    obj_h = 95.25
                                else:             # Simulation Bot
                                    obj_h = 60.0
                            else:
                                cam_h = 3000.0
                                if aruco_id in (0, 2, 4): # Real Bot
                                    obj_h = 111.25
                                else:             # Real Crate
                                    obj_h = 51.0

                            # Apply perspective correction based on height
                            scale = obj_h / cam_h
                            x -= scale * (x - center_mm)
                            y -= scale * (y - center_mm)

                            # --- YAW CALCULATION ---
                            rvecs, _, _ = cv2.aruco.estimatePoseSingleMarkers(
                                np.array([marker_corners]), self.bots_marker_length, 
                                self.camera_matrix, self.dist_coeffs
                            )
                            rmat, _ = cv2.Rodrigues(rvecs[0][0])
                            yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

                            # --- DATA PACKAGING ---
                            dict_key = int(aruco_id) 
                            pose_data = {'id': dict_key, 'x': x, 'y': y, 'yaw': yaw}
                            
                            if aruco_id in (0, 2, 4):
                                bot_poses[dict_key] = pose_data
                            elif (self.arena_min <= x <= self.arena_max) and (self.arena_min <= y <= self.arena_max):
                                crate_poses[dict_key] = pose_data

                    self.publish_crate_poses(list(crate_poses.values()))
                    self.publish_bot_poses(list(bot_poses.values()))
                    self.publish_camera_debug(bot_poses, crate_poses)

                    # Draw HUD using the advanced visualization
                    self.draw_hud(undistorted_image, ids, refined_corners, bot_poses, crate_poses)
            
            cv2.imshow('Detected Markers', undistorted_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def publish_crate_poses(self, poses):
        poses_msg = Poses2D()
        for pose_data in poses:
            crate_pose = Pose2D()
            crate_pose.id = pose_data['id']
            crate_pose.x = pose_data['x']
            crate_pose.y = pose_data['y']
            crate_pose.w = pose_data['yaw']
            poses_msg.poses.append(crate_pose)
        self.crate_poses_pub.publish(poses_msg)

    def publish_bot_poses(self, poses):
        poses_msg = Poses2D()
        for pose_data in poses:
            bot_pose = Pose2D()
            bot_pose.id = pose_data['id']
            bot_pose.x = pose_data['x']
            bot_pose.y = pose_data['y']
            bot_pose.w = pose_data['yaw']
            poses_msg.poses.append(bot_pose)
        self.bot_poses_pub.publish(poses_msg)

    def publish_camera_debug(self, bot_poses, crate_poses):
        now = self.get_clock().now().nanoseconds / 1e9
        packet = {
            "timestamp": now,
            "bots": bot_poses,
            "crates": crate_poses
        }
        msg = String()
        msg.data = json.dumps(packet)
        self.camera_debug_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    pose_detector = PoseDetector()
    try:
        rclpy.spin(pose_detector)
    except KeyboardInterrupt:
        pass
    finally:
        pose_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()