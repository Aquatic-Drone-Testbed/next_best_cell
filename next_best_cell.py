import time
import math
import numpy as np

import re
import os

def select_next_frontier(pose, graph,
                         lambda_coeff=1.0,
                         theta_weight=1.0,
                         free_threshold=0.5,
                         debug_mode=False,
                         max_runtime_ms=5000):
    """
    Given the current robot pose and an occupancy graph, select the next best frontier cell to explore.
    Inspired by "A Frontier-Based Approach for Autonomous Exploration"
        Brian Yamauchi, Naval Research Laboratory 1997

    Parameters:
        pose: tuple (x, y, theta) giving the robot's current grid coordinates (x=row, y=col) and heading in radians.
        graph: numpy array of shape (500, 500, 2), where graph[i,j,0] = count of observations as occupied (non-empty),
               and graph[i,j,1] = count of observations in total for cell (i,j).
        lambda_coeff: weight balancing information gain vs cost in the utility function.
        theta_weight: additional weight on turning cost in the utility function.
        free_threshold: occupancy probability below which a cell is considered free (traversable).
        debug_mode: if True, return (best_cell, info_dict) with utility components; if False, return just best_cell.
        max_runtime_ms: time budget for the computation (not strictly enforced in this implementation).

    Returns:
        If debug_mode is False: (i, j) coordinates of the best frontier cell to explore next (or None if no frontier).
        If debug_mode is True: a tuple (best_cell, info) where info is a dict with keys 'utility', 'gain',
                               'distance_cost', 'turning_cost', 'elapsed_ms'.
        If no frontier is found, returns None (and an empty dict in debug_mode).
    """
    # Start the timer
    start_time = time.perf_counter()

    # Unpack pose
    x, y, theta = pose  # assuming x=grid row index, y=grid column index

    # Determine which cells are known free, known occupied, or unknown
    obs_counts = graph[:, :, 1]
    occ_counts = graph[:, :, 0]
    known_mask = obs_counts > 0  # cells that have been observed at least once
    unknown_mask = obs_counts == 0  # cells never observed (unmapped)
    # Compute occupancy probability for known cells
    occ_prob = np.zeros_like(occ_counts, dtype=float)
    occ_prob[known_mask] = occ_counts[known_mask] / obs_counts[known_mask]
    # Free cells = known and occupancy probability below threshold
    free_mask = known_mask & (occ_prob < free_threshold)

    # Identify frontier cells (free cell with at least one unknown neighbor)
    # We check all 8 neighbors for unknown status. We'll use array slicing for efficiency.
    u = unknown_mask  # alias for readability
    neighbors_unknown = np.zeros_like(u, dtype=bool)
    # Check neighbor in each of 8 directions (with boundary checks)
    if u.shape[0] > 1:
        neighbors_unknown[:-1, :] |= u[1:, :]  # neighbor below
        neighbors_unknown[1:, :] |= u[:-1, :]  # neighbor above
    if u.shape[1] > 1:
        neighbors_unknown[:, :-1] |= u[:, 1:]  # neighbor right
        neighbors_unknown[:, 1:] |= u[:, :-1]  # neighbor left
    if u.shape[0] > 1 and u.shape[1] > 1:
        neighbors_unknown[:-1, :-1] |= u[1:, 1:]  # bottom-right neighbor
        neighbors_unknown[:-1, 1:] |= u[1:, :-1]  # bottom-left neighbor
        neighbors_unknown[1:, :-1] |= u[:-1, 1:]  # top-right neighbor
        neighbors_unknown[1:, 1:] |= u[:-1, :-1]  # top-left neighbor
    # Frontier mask: free cell with any unknown neighbor
    frontier_mask = free_mask & neighbors_unknown

    # If no frontiers found, return None or empty result
    if not np.any(frontier_mask):
        if debug_mode:
            return None, {}
        else:
            return None

    # Cluster frontier cells into connected regions (8-connected components)
    frontier_indices = np.transpose(np.nonzero(frontier_mask))
    visited = np.zeros_like(frontier_mask, dtype=bool)
    clusters = []
    # Directions for 8-neighbor connectivity
    neighbor_deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for (ci, cj) in frontier_indices:
        # Check budget before starting each new cluster
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > max_runtime_ms:
            # Abort clustering and go straight to evaluation of what we have
            break

        if visited[ci, cj]:
            continue  # skip if already part of a cluster
        # Start a new cluster BFS/DFS from (ci, cj)
        cluster_cells = []
        stack = [(ci, cj)]
        visited[ci, cj] = True
        while stack:
            i, j = stack.pop()
            cluster_cells.append((i, j))
            # Explore all neighbors of this frontier cell
            for (di, dj) in neighbor_deltas:
                ni, nj = i + di, j + dj
                if 0 <= ni < frontier_mask.shape[0] and 0 <= nj < frontier_mask.shape[1]:
                    if frontier_mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        stack.append((ni, nj))
        clusters.append(cluster_cells)

    # Evaluate each cluster's utility
    best_score = -math.inf
    best_cell = None
    best_info = None
    for cluster_cells in clusters:
        # Compute information gain (number of frontier cells in this cluster)
        gain = len(cluster_cells)
        # Compute cluster centroid
        ci_mean = sum(i for (i, j) in cluster_cells) / gain
        cj_mean = sum(j for (i, j) in cluster_cells) / gain
        # Distance from current position to cluster centroid
        dx = cj_mean - y
        dy = ci_mean - x
        distance = math.hypot(dx, dy)
        # Heading to centroid and turning angle difference from current theta
        target_angle = math.atan2(dy, dx)
        delta_theta = target_angle - theta
        # Normalize delta_theta to the range [-pi, pi] for smallest rotation
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
        turning_cost = abs(delta_theta)
        # Utility = gain - lambda * (distance + theta_weight * turning_cost)
        utility = gain - lambda_coeff * (distance + theta_weight * turning_cost)
        # Check if this cluster is the best so far
        if utility > best_score:
            best_score = utility
            # Choose a representative frontier cell in this cluster as the goal.
            # We pick the frontier cell closest to the centroid (could also pick closest to robot).
            goal_cell = min(cluster_cells, key=lambda cell: (cell[0] - ci_mean) ** 2 + (cell[1] - cj_mean) ** 2)
            best_cell = goal_cell
            best_info = {
                'utility': utility,
                'gain': gain,
                'distance_cost': distance,
                'turning_cost': turning_cost,
            }

    # Return the result
    if debug_mode:
        # Track time elapsed for debug info
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        best_info['elapsed_ms'] = elapsed_ms

        return best_cell, best_info
    else:
        return best_cell
        
pattern = r'(\d+)\t\((\d+),(\d+)\)\t(-?\d+\.\d+)'

with open('/home/seamate1/ControlStationFiles/NAV2_USV/map/current_pose_estimation.txt', 'r') as file:
    content = file.read()

matches = re.findall(pattern, content)

pose_list = [(int(a), (float(b)), (float(c)), float(d)*math.pi/180.0) for a, b, c, d in matches]
pose = (pose_list[0][1],pose_list[0][2],pose_list[0][3])

np_map = np.load('/home/seamate1/ControlStationFiles/NAV2_USV/map/curr_stitched_map_times_counted.npy')             # shape (H, W)
np_map = np.flipud(np_map)

np2_map = np.load('/home/seamate1/ControlStationFiles/NAV2_USV/map/curr_stitched_map_times_viewed.npy')             # shape (H, W)
np2_map = np.flipud(np2_map)

stacked_array = np.stack((np_map, np2_map), axis=2)

print(pose)
#print(stacked_array)
#print(select_next_frontier(pose, stacked_array))
goal = select_next_frontier(pose, stacked_array)
goal = (float(goal[0]),float(goal[1]))
position_str = f"{goal}"
print(position_str)

latest_pose = f"{0}\t{position_str}\t{0.0}\n"
if latest_pose:
    with open('/home/seamate1/ControlStationFiles/NAV2_USV/map/goal_pose.txt', 'w') as goal_pose_f:
        goal_pose_f.write(latest_pose)
