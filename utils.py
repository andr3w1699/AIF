import math
from collections import deque
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

action_map = {
    "N": 0,
    "E": 1,
    "S": 2,
    "W": 3,
    "NE": 4,
    "SE": 5,
    "SW": 6,
    "NW": 7
}

directions = ["UP", "RIGHT", "DOWN", "LEFT", "UP_RIGHT", "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT"]


def get_player_location(game_map: np.ndarray, symbol: str = "@") -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return int(x[0]), int(y[0])


def get_stairs_location(game_map: np.ndarray, symbol: str = ">") -> tuple[int, int] | None:
    x, y = np.where(game_map == ord(symbol))
    if x.size == 0 or y.size == 0:
        return None  # No stairs found
    return int(x[0]), int(y[0])


def is_floor_tile(lines, x, y):
    if 0 <= y < len(lines) and 0 <= x < len(lines[y]):
        return lines[y][x] == '.'
    return False


def is_wall(position_element) -> bool:
    obstacles = "|-}"
    return chr(int(position_element)) in obstacles


def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int], avoid_stairs=False,
                    allow_diagonals=True) -> List[
    Tuple[int, int]]:
    """
        Returns a list of valid moves from the current position on the game map.

        :param game_map: The map of the game, with each cell representing a tile.
        :param current_position: The current (x, y) position.
        :param avoid_stairs: If True, treat stairs ('>') as obstacle.
        :param allow_diagonals: If True, include diagonal moves.

        :return: List of valid (x, y) positions that can be moved to.
    """
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position
    # North
    if y - 1 > 0 and not is_wall(game_map[x, y - 1]):
        if not (avoid_stairs and game_map[x, y - 1] == ord('>')):
            valid.append((x, y - 1))

    # East
    if x + 1 < x_limit and not is_wall(game_map[x + 1, y]):
        if not (avoid_stairs and game_map[x + 1, y] == ord('>')):
            valid.append((x + 1, y))

    # South
    if y + 1 < y_limit and not is_wall(game_map[x, y + 1]):
        if not (avoid_stairs and game_map[x, y + 1] == ord('>')):
            valid.append((x, y + 1))

    # West
    if x - 1 > 0 and not is_wall(game_map[x - 1, y]):
        if not (avoid_stairs and game_map[x - 1, y] == ord('>')):
            valid.append((x - 1, y))

    if allow_diagonals:
        # Needs to check if the diagonal move is valid. If the two adjacent tiles are not walls, then the diagonal move is valid.
        # North-East
        if (y - 1 > 0 and x + 1 < x_limit and
                not is_wall(game_map[x + 1, y - 1]) and
                not is_wall(game_map[x, y - 1]) and
                not is_wall(game_map[x + 1, y])):
            if not (avoid_stairs and game_map[x + 1, y - 1] == ord('>')):
                valid.append((x + 1, y - 1))
        # North-West
        if (y - 1 > 0 and x - 1 > 0 and
                not is_wall(game_map[x - 1, y - 1]) and
                not is_wall(game_map[x, y - 1]) and
                not is_wall(game_map[x - 1, y])):
            if not (avoid_stairs and game_map[x - 1, y - 1] == ord('>')):
                valid.append((x - 1, y - 1))

        # South-East
        if (y + 1 < y_limit and x + 1 < x_limit and
                not is_wall(game_map[x + 1, y + 1]) and
                not is_wall(game_map[x, y + 1]) and
                not is_wall(game_map[x + 1, y])):
            if not (avoid_stairs and game_map[x + 1, y + 1] == ord('>')):
                valid.append((x + 1, y + 1))

        # South-West
        if (y + 1 < y_limit and x - 1 > 0 and
                not is_wall(game_map[x - 1, y + 1]) and
                not is_wall(game_map[x, y + 1]) and
                not is_wall(game_map[x - 1, y])):
            if not (avoid_stairs and game_map[x - 1, y + 1] == ord('>')):
                valid.append((x - 1, y + 1))

    return valid


def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> List[int]:
    """
       Converts a path (list of positions) into a list of action indices based on movement direction.

       :param start: The starting position as a tuple (y, x).
       :param path: A list of positions (tuples) representing the path to follow.
       :return: A list of action indices corresponding to the moves between positions.
       """
    actions = []
    y_s, x_s = start

    for (y, x) in path:
        if x_s == x:
            if y_s > y:
                actions.append(action_map["N"])  # Up
            else:
                actions.append(action_map["S"])  # Down
        elif y_s == y:
            if x_s > x:
                actions.append(action_map["W"])  # Left
            else:
                actions.append(action_map["E"])  # Right
        else:
            if y_s > y and x_s > x:
                actions.append(action_map["NW"])
            elif y_s > y and x_s < x:
                actions.append(action_map["NE"])
            elif y_s < y and x_s > x:
                actions.append(action_map["SW"])
            else:
                actions.append(action_map["SE"])

        y_s = y
        x_s = x

    return actions


def chebyshev_distance(point1: Tuple[int, int], point2: Tuple[int, int], **kwargs) -> int:
    x1, y1 = point1
    x2, y2 = point2
    return max(abs(x1 - x2), abs(y1 - y2))


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int], **kwargs) -> int:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def cached_bfs(game_map, start, goal, path_cache):
    key = (start, goal)
    if key in path_cache:
        return path_cache[key]
    dist = bfs_path_length(game_map, start, goal)
    path_cache[key] = dist
    return dist


def bfs_path_length(game_map, point1, point2) -> int | float:
    """Return shortest path length between start and goal, accounting for walls."""
    if point1 == point2:
        return 0
    rows, cols = game_map.shape
    visited = set()
    queue = deque([(point1, 0)])
    visited.add(point1)

    while queue:
        (x, y), dist = queue.popleft()
        for nx, ny in get_valid_moves(game_map, (x, y)):
            if (nx, ny) == point2:
                return dist + 1
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return float('inf')  # no path


def randomize_apple_positions(
        map_str, min_x, min_y, max_x, max_y, num_apple, seed=None
):
    """
    Randomizes apple positions in a given map string.

    :param map_str: The map in string representation.
    :param min_x: Minimum x-coordinate for apple placement.
    :param min_y: Minimum y-coordinate for apple placement.
    :param max_x: Maximum x-coordinate for apple placement.
    :param max_y: Maximum y-coordinate for apple placement.
    :param num_apple: Number of apple piles to place.
    :param seed: Optional seed for random number generation.

    :return: A list of tuples representing the positions of the apple piles.

    """

    import random

    if seed is not None:
        random.seed(seed)

    apple_positions = []
    lines = [line.rstrip() for line in map_str.strip().split('\n')]
    # print(f"Placing {num_apple} apples between ({min_x}, {min_y}) and ({max_x}, {max_y})")
    while len(apple_positions) < num_apple:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        if (x, y) not in apple_positions:
            if is_floor_tile(lines, x, y) and lines[y][x] != '%':
                apple_positions.append((x, y))
            else:
                continue

    return apple_positions


def print_path_on_map(game_map: np.ndarray, path: List[Tuple[int, int]]):
    path_set = set(path)
    for y in range(game_map.shape[0]):
        row = ""
        for x in range(game_map.shape[1]):
            pos = (y, x)
            char = chr(int(game_map[y, x]))
            if pos == path[0]:
                row += "@"
            elif pos == path[-1] and char == '>':
                row += "!"
            elif pos in path_set:
                if char == '%':
                    row += "A"  # collected apple
                elif char == '>':
                    row += ">"
                else:
                    row += "*"  # part of the path
            else:
                row += char
        print(row)


def simulate_path(path, game_map, actions):
    """ Simulate the path on the game map and print the actions taken.
    """
    print("Simulating path:", path)
    print("Actions to take:", list((map(lambda x: directions[x], actions))))
    # check how much apple is collected in the path
    apple_collected = []
    for (x, y) in path:
        if chr(game_map[x, y]) == '%' and (x, y) not in apple_collected:
            apple_collected.append((x, y))
    apple_collected = len(apple_collected)
    print("Apple collected in the path:", apple_collected)
    print("Expected Reward: ", 1 + apple_collected * 0.75 - 0.1 * len(path))
    print_path_on_map(game_map, path)


def save_images_as_video(images, save_dir: str, file_name: str, fps):
    # Create directory if it doesn't exist
    import os
    if not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir)

    from matplotlib import animation

    fig = plt.figure(figsize=(10, 10))
    frames = [[plt.imshow(img, animated=True)] for img in images]
    ani = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True)

    filename = file_name + '.gif'
    print(f"Saving video to {os.path.join(save_dir, filename)}")
    ani.save(os.path.join(save_dir, filename), writer='ffmpeg', fps=fps)


def plot_3d_surfaces(
        agg_df,
        x_col,
        y_col,
        z1_col,
        z2_col,
        z1_label="Metric 1",
        z2_label="Metric 2",
        z1_title=None,
        z2_title=None,
        figsize=(14, 6)
):
    """
    Creates two 3D surface plots for given metrics over specified parameters.

    Parameters:
    - agg_df: Aggregated DataFrame (grouped and averaged)
    - x_col: Column name for X-axis (e.g., 'beam_width', 'weight')
    - y_col: Column name for Y-axis (e.g., 'apple_reward', 'apple_bonus')
    - z1_col: First metric column for Z-axis (e.g., 'avg_reward')
    - z2_col: Second metric column for Z-axis (e.g., 'avg_path_length')
    - z1_label: Z-axis label for the first plot
    - z2_label: Z-axis label for the second plot
    - z1_title: Title for the first plot
    - z2_title: Title for the second plot
    - figsize: Size of the figure
    """
    x_vals = sorted(agg_df[x_col].unique())
    y_vals = sorted(agg_df[y_col].unique())

    X, Y = np.meshgrid(x_vals, y_vals)
    Z1 = np.full_like(X, np.nan, dtype=float)
    Z2 = np.full_like(X, np.nan, dtype=float)

    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            row = agg_df[(agg_df[x_col] == x_val) & (agg_df[y_col] == y_val)]
            if not row.empty:
                Z1[i, j] = row[z1_col].values[0]
                Z2[i, j] = row[z2_col].values[0]

    fig = plt.figure(figsize=figsize)

    # Plot 1
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap="viridis", edgecolor='k', alpha=0.85)
    ax1.set_title(z1_title or f"{z1_label} vs {x_col} & {y_col}")
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Plot 2
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap="plasma", edgecolor='k', alpha=0.85)
    ax2.set_title(z2_title or f"{z2_label} vs {x_col} & {y_col}")
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()
