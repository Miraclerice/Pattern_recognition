# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import random

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import os

"""
概念
    深度优先搜索算法（Depth First Search）：英文缩写为 DFS，是一种用于搜索树或图结构的算法。
    深度优先搜索算法采用了回溯思想，从起始节点开始，沿着一条路径尽可能深入地访问节点，直到无法继续前进时为止，
    然后回溯到上一个未访问的节点，继续深入搜索，直到完成整个搜索过程。
步骤
    1. 选择起始节点u,并将其标记为已访问
    2. 检查当前节点是否为目标节点
    3. 如果当前节点u是目标节点，直接返回结果
    4. 如果当前节点u不是目标节点，则遍历当前节点u的所有未访问邻接节点v
    5. 如果v未被访问，则将v标记为已访问，从节点v出发继续进行深度优先搜索（递归）
    6. 如果v没有维访问的相邻节点，回溯到上一节点，继续搜索其他路径
    7. 重复2-6直到遍历整个图或找到目标节点为止
"""

"""
概念：
    广度优先搜索算法（Breadth First Search）：英文缩写为 BFS，又译作宽度优先搜索 / 横向优先搜索，是一种用于搜索树或图结构的算法。
    广度优先搜索算法从起始节点开始，逐层扩展，先访问离起始节点最近的节点，后访问离起始节点稍远的节点。以此类推，直到完成整个搜索过程。
步骤
    1. 选择起始节点u, 放入队列，标记为已访问
    2. 从队列中取出节点， 访问它并将其所有的未访问邻接节点v放入队列中
    3. 标记节点v为已访问， 避免重复访问
    4. 重复2-3直到队列为空或找到目标节点
"""


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
                print(len(np.where(maze == 1)[0]))
                print(self.obstacle_coverage * maze.shape[0] * maze.shape[1])
                print(len(np.where(maze == 1)[0]) == self.obstacle_coverage * maze.shape[0] * maze.shape[1])
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

        # TODO: test search function, please comment out after debugging
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


if __name__ == '__main__':
    maze = Maze(size=(50, 50), obstacle_coverage=0.4, start=(0, 0), end=(49, 49))()
    Maze.plot_maze(maze, (0, 0), (49, 49), path='./test/img/', filename='maze.png')
