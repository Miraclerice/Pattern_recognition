# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import os
import pickle
import sys
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class Logger(object):
    def __init__(self, save_dir, filename="res.log"):
        self.terminal = sys.stdout
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log = open(os.path.join(save_dir, filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Maze(object):
    """
    Generates a maze of the specified size
    :param
    size: (int, int) size of the maze
    obstacle_coverage: float The proportion of obstacles to the maze
    start: (int, int) start grid
    end: (int, int) end grid
    seed: int random seed, Since the maze requires a feasible solution, it does not work
    ratio: float The proportion of randomly removed obstacles
    """

    def __init__(self, size, obstacle_coverage, start, end, ratio=0.05, seed=42):
        self.size = size
        self.obstacle_coverage = obstacle_coverage + ratio
        self.start = start
        self.end = end
        self.seed = seed
        self.ratio = ratio

    def __call__(self, *args, **kwargs):
        """Gets the maze of the object"""
        return self.init_maze()

    def init_maze(self):
        """
        Generate a maze with passable paths
        """
        while True:
            maze = self._generate_maze()
            visited = np.zeros(self.size, dtype=bool)
            visited[self.start] = True

            # Depth First Search with recursion
            # if self.dfs_recursion(maze, visited, self.start, self.end):
            #     self.random_rm_obstacle(maze, self.obstacle_coverage, visited, self.ratio)
            #     self.pkl_save(maze, 'maze.pkl')
            #     return maze

            # Depth First Search without loop
            if self.dfs_loop(maze, visited, self.start, self.end):
                self.random_rm_obstacle(maze, self.ratio)
                self.pkl_save(maze, 'maze.pkl')
                return maze

            # Breadth First Search
            # if self.bfs(maze, visited, self.start, self.end):
            #     self.random_rm_obstacle(maze, self.obstacle_coverage, visited, self.ratio)
            #     self.pkl_save(maze, 'maze.pkl')
            #     return maze

    def _generate_maze(self):
        """Randomly generate a maze with obstacles"""
        size = self.size
        maze = np.zeros(size, dtype=np.int32)
        # Random seeds cannot be specified. An available path must be generated multiple times.
        # If multiple parameters are required, save them
        # np.random.seed(self.seed)
        obstacle = np.random.choice(size[0] * size[1], int(self.obstacle_coverage * size[0] * size[1]), replace=False)
        obstacle_pos = [(pos // size[1], pos % size[0]) for pos in obstacle]

        for pos in obstacle_pos:
            maze[pos] = 1

        # test search function, please comment out after debugging
        # self.plot_maze(maze, start=self.start, end=self.end, path='./test/img', filename='maze_init.png')
        return maze

    @staticmethod
    def is_valid_path(maze, visited, row, col):
        # Determines whether the current point (row, col) is within the maze range, and the grid is unobstructed and not visited
        return maze.shape[0] > row >= 0 and maze.shape[1] > col >= 0 == maze[row, col] and not visited[row, col]

    def dfs_recursion(self, maze, visited, start, end):
        # Find the feasible path recursively
        if start == end:
            return True
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # It doesn't really matter, because it doesn't affect the outcome
        # random.shuffle(directions)
        for d_row, d_col in directions:
            new_row, new_col = start[0] + d_row, start[1] + d_col
            if self.is_valid_path(maze, visited, new_row, new_col):
                visited[new_row, new_col] = True
                if self.dfs_recursion(maze, visited, (new_row, new_col), end):
                    return True
        return False

    def dfs_loop(self, maze, visited, start, end):
        # use stack implementation, first in, last out
        stack = deque([[start, [start]]])
        while stack:
            cur, route = stack.pop()
            if cur == end:
                cmap = plt.cm.colors.ListedColormap(['#FFFFFF', '#000000', '#A9A9E6', '#E6A9A9', '#1AFF1A'])
                self.plot_maze(maze.copy(), start, end, route, cmap=cmap, path='./test/img', filename='maze_res.svg')
                return True
            # If not shuffle, the preorder traversal (up, down, left and right, that is, the reverse order of the direction list)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            # random.shuffle(directions)
            for d_row, d_col in directions:
                new_row, new_col = cur[0] + d_row, cur[1] + d_col
                if self.is_valid_path(maze, visited, new_row, new_col):
                    visited[new_row, new_col] = True
                    stack.append([(new_row, new_col), route + [(new_row, new_col)]])
        return False

    def bfs(self, maze, visited, start, end):
        # use queue implementation, first in, first out
        queue = deque([[start, [start]]])
        while queue:
            # pop the first element, it can be understood as hierarchical traversal
            cur, route = queue.popleft()
            if cur == end:
                cmap = plt.cm.colors.ListedColormap(['#FFFFFF', '#000000', '#A9A9E6', '#E6A9A9', '#1AFF1A'])
                self.plot_maze(maze.copy(), start, end, route, cmap=cmap, path='./test/img', filename='maze_res.svg')
                return True
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            # random.shuffle(directions)
            for d_row, d_col in directions:
                new_row, new_col = cur[0] + d_row, cur[1] + d_col
                if self.is_valid_path(maze, visited, new_row, new_col):
                    visited[new_row, new_col] = True
                    queue.append([(new_row, new_col), route + [(new_row, new_col)]])
        return False

    @staticmethod
    def random_rm_obstacle(maze, ratio):
        """
         More feasible paths can be obtained by randomly removing obstacles
         while keeping the proportion of obstacles unchanged
        """
        size = int(maze.shape[0] * maze.shape[1] * ratio)
        for i in range(size):
            while True:
                row = np.random.randint(maze.shape[0])
                col = np.random.randint(maze.shape[1])
                if maze[row, col] == 1:
                    maze[row, col] = 0
                    break

    @staticmethod
    def pkl_save(matrix, save_path):
        """Save the maze grid matrix with the pickle type"""
        with open(save_path, 'wb') as f:
            pickle.dump(matrix, f)

    @staticmethod
    def pkl_load(path):
        """Load the maze grid matrix with the pickle type"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def plot_maze(maze, start, end, route=None, cmap=None, path='./img', filename='maze.svg'):
        """
        Plot maze
        """
        assert isinstance(start, tuple), f'start must be tuple, but got {type(start)}'
        assert isinstance(end, tuple), f'end must be tuple, but got {type(end)}'
        # Modify the pixel values for the start and end points
        maze[start] = 2
        maze[end] = 3
        if route is not None:
            assert cmap is not None, 'If you add routes, you must add color mapping'
            for row, col in route[1:-1]:
                maze[row, col] = 4

        if cmap is None:
            # Color mappings correspond to pixel values one by one
            cmap = plt.cm.colors.ListedColormap(['#FFFFFF', '#000000', '#A9A9E6', '#E6A9A9'])
        plt.figure(figsize=(8, 8))
        plt.imshow(maze, cmap=cmap)
        for row in range(len(maze)):
            plt.axhline(row - 0.5, color='gray', linestyle='-', lw=0.5)
        for col in range(len(maze[0])):
            plt.axvline(col - 0.5, color='gray', linestyle='-', lw=0.5)

        # Hide the x-axis and y-axis scale labels in the chart
        plt.xticks([])
        plt.yticks([])

        # Set axis properties
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)
            spine.set_color('#818181')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}/{filename}', bbox_inches='tight')
        plt.show()


def init_se(size, mode='random', seed=42):
    """
    Specify a start and end point
    :param
    size: maze size
    mode: 'random' or 'constant'
    seed: random seed
    """
    assert mode in ['random', 'constant'], 'mode must be random or constant'
    if mode == 'constant':
        return [(0, 0), (size[0] - 1, size[1] - 1)]
    else:
        # TODO: The two coordinate values cannot be too close
        np.random.seed(seed)
        while True:
            s_row, s_col = np.random.randint(size), np.random.randint(size)
            e_row, e_col = np.random.randint(size), np.random.randint(size)
            if s_row != e_row or s_col != e_col:
                return [(s_row, s_col), (e_row, e_col)]


class ACO(object):
    """
    The ant colony algorithm is used to solve the optimal path of the maze
    :param
    maze: (n, m) maze matrix
    start: start point
    end: end point
    ants_num: number of ants
    alpha: importance of pheromone
    beta: heuristic factor
    rho: pheromone evaporation coefficient
    q: pheromones increase strength
    epoch: number of iterations
    eta: heuristic factor
    tau: pheromone matrix
    tabu: tabu table
    """

    def __init__(self, maze, start, end, ants_num=50, alpha=1., beta=5, rho=0.1, q=100, epoch=100):
        self.maze = maze
        self.start = start
        self.end = end
        self.ants_num = ants_num
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.epoch = epoch
        self.tau = np.ones_like(maze)
        self.tabu = []
        # The maze is a 50 by 50 grid, each with a distance of 1
        # whether or not to be specific needs further consideration
        self.eta = np.ones_like(maze)

        # Record the average distance, shortest distance, and shortest route for each iteration
        self.route_best = []
        self.dist_best = np.zeros(epoch)
        self.dist_mean = np.zeros(epoch)

    def train(self):
        save_dir, filename = './log', 'res.log'
        sys.stdout = Logger(save_dir=save_dir, filename=filename)
        start_time = time.time()
        for i in range(self.epoch):
            ants_dist = np.zeros(self.ants_num)
            process_bar = tqdm(range(self.ants_num))
            for j in process_bar:
                process_bar.set_description(f'[Iteration: {i + 1:0>3d}, Ant: {j + 1:0>3d}]')
                start, end = self.start, self.end
                route = self.move(start, end)

                # Based on route, calculate the total path distance of the current ant
                self.calc_dist(j, ants_dist, route)
                self.tabu.append(route)

            # Record the average distance, shortest distance, and shortest route for each iteration
            self.record_res(i, ants_dist)

            # Update the pheromone matrix
            self.update_tau(i, ants_dist)

            # Clear tabu table
            self.tabu = []
            process_bar.close()
        end_time = time.time()
        # Save the optimal result
        res = self.result_save(end_time - start_time)
        # Plot the result
        self.plot(res, img_dir='img')
        sys.stdout = sys.__stdout__

    def move(self, start, end):
        """The ants travel through different grids in the maze until they reach the end"""
        # record the visited grids, route may be different from tabu table
        route = [start]
        while start != end:
            # Further optimization, did not achieve the ideal result, may be wrong pheromone, path selection
            directions = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
            prob = np.zeros(len(directions))
            for i, (d_row, d_col) in enumerate(directions):
                next_row, next_col = start[0] + d_row, start[1] + d_col
                try:
                    if self.is_valid_path(next_row, next_col):
                        prob[i] = self.tau[next_row, next_col] ** self.alpha * self.eta[next_row, next_col] ** self.beta
                except IndexError as e:
                    print(next_row, next_col)
                    print(route, e)

            # The probability of an obstacle (0) needs to be removed, dim=1 [0]
            nz_idx = np.nonzero(prob)[0]
            prob = prob[nz_idx]
            directions = directions[nz_idx]
            # roulette wheel selection
            prob = prob / prob.sum()
            prob_sum = np.cumsum(prob)

            r = np.random.rand()
            idx = np.where(prob_sum > r)[0][0]
            next_grid = (start[0] + directions[idx][0], start[1] + directions[idx][1])
            if next_grid in route:
                slice_idx = route.index(next_grid)
                route = route[:slice_idx + 1]
            else:
                route.append(next_grid)
            start = next_grid

        return route

    def is_valid_path(self, row, col):
        """Determine whether the coordinate points are within the maze grid and whether they are passable grids"""
        return maze.shape[0] > row >= 0 and maze.shape[1] > col >= 0 == self.maze[row, col]

    def record_res(self, cur_iter, ants_dist):
        """Record the average distance, shortest distance, and shortest route for each iteration"""
        self.dist_mean[cur_iter] = np.mean(ants_dist)
        self.dist_best[cur_iter] = np.min(ants_dist)
        print(f'Iter: {cur_iter + 1:0>3d}, Mean: {self.dist_mean[cur_iter]:.2f}, Best: {self.dist_best[cur_iter]:.2f}')
        self.route_best.append(self.tabu[np.argmin(ants_dist)])

    def update_tau(self, cur_iter, ants_dist):
        """Update the pheromone matrix"""
        pheromone = np.zeros(self.maze.shape)
        for i, route in enumerate(self.tabu):
            for j in range(len(route) - 1):
                pheromone[route[j + 1]] += self.q / ants_dist[i]
        self.tau = (1 - self.rho) * self.tau + pheromone

    @staticmethod
    def calc_dist(cur_ant, ants_dist, route):
        """Calculate the total path distance of the current ant"""
        ants_dist[cur_ant] = len(route) - 1

    def plot(self, res, img_dir='img'):
        """Plot the maze and convergence curve"""
        dist_best, idx, route_best = res
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        cmap = plt.cm.colors.ListedColormap(['#FFFFFF', '#000000', '#A9A9E6', '#E6A9A9', '#1AFF1A'])
        # Plot the optimal result
        Maze.plot_maze(self.maze, self.start, self.end, route=route_best, cmap=cmap, path=img_dir,
                       filename='maze_res.svg')

        # Plot the convergence curve
        plt.figure(dpi=500)
        plt.plot(range(self.epoch), self.dist_mean, 'b', range(self.epoch), self.dist_best, 'r')
        plt.legend(['Average distance', 'Shortest distance'])
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Convergence curve')
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.savefig(img_dir + '/Convergence_curve.svg', bbox_inches='tight')
        plt.show()

    def result_save(self, consume_time):
        """Save the results to log file"""
        dist_best, idx = np.min(self.dist_best), np.argmin(self.dist_best)
        route_best = self.route_best[idx]
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f'Total iterations: {self.epoch}')
        print(f'Saturation iteration: {idx + 1}')
        print(f'Best route: {route_best}')
        print(f'Best distance: {dist_best}')
        print(f'Consume time: {consume_time:.2f}s')
        return dist_best, idx, route_best


if __name__ == '__main__':
    maze_size = [50, 50]
    obstacle_coverage = 0.45
    start, end = init_se(maze_size, mode='constant')
    # print(start, end)
    maze = Maze(maze_size, obstacle_coverage, start, end)()
    # maze = Maze.pkl_load('maze.pkl')
    # Maze.plot_maze(maze.copy(), start, end)
    # print(maze)
    ACO(maze, start, end, 30, 1, 5, 0.1, 100, 10).train()
    print(len(np.where(maze == 1)[0]) == obstacle_coverage * maze_size[0] * maze_size[1])
