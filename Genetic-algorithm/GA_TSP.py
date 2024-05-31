# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import os
import sys
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyproj import Geod
from argparse import ArgumentParser


class Logger(object):
    def __init__(self, save_dir, filename="log.txt"):
        self.terminal = sys.stdout
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log = open(os.path.join(save_dir, filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def central_angle_cos(coord1, coord2):
    """Cosine theorem of the sphere to calculate the central angle"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [coord1[0], coord1[1], coord2[0], coord2[1]])
    dlon = lon2 - lon1
    theta = np.arccos(np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1) * np.sin(lat2))
    return theta


def central_angle_hav(coord1, coord2):
    """Haversine formula to calculate the central angle"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [coord1[0], coord1[1], coord2[0], coord2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    theta = 2 * np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return theta


def central_angle_proj(coord1, coord2):
    """pyproj.Geod to calculate the central angle"""
    return Geod(ellps='WGS84').inv(coord1[0], coord1[1], coord2[0], coord2[1])[2]


def cal_distance(coord1, coord2, mode, radius=6371):
    # Cosine theorem of the sphere
    if mode == 'cos':
        return radius * central_angle_cos(coord1, coord2)
    # haversine formula
    elif mode == 'hav':
        return radius * central_angle_hav(coord1, coord2)
    # pyproj.Geod
    elif mode == 'proj':
        return central_angle_proj(coord1, coord2) / 1000
    else:
        raise ValueError('Wrong mode!')


def init_route(n_route, n_city, seed=42):
    route_mat = np.zeros((n_route, n_city), dtype=np.int32)
    # np.random.seed(seed=seed)
    for i in range(n_route):
        route_mat[i] = np.random.permutation(n_city)
    return route_mat


def selection(all_dist, route_mat):
    """
    select routes by probability
    :param all_dist: the total distance of each route
    :param route_mat: route matrix for all populations
    :return: next generation of route matrix
    """
    # fitness
    fit = 1 / all_dist
    prob = fit / np.sum(fit)
    # calculate cumulative probability
    prob_sum = np.cumsum(prob)
    # index list of selected routes
    select_idx = []
    for i in range(len(route_mat)):
        r = np.random.rand()
        for j in range(len(route_mat)):
            # Initial index single value interval (0, +)
            if r < prob_sum[0]:
                select_idx.append(0)
                break
            # Interval by interval judgment
            elif prob_sum[j] < r <= prob_sum[j + 1]:
                select_idx.append(j + 1)
                break
    next_gen = route_mat[select_idx, :]
    return next_gen


def crossover(route_mat, n, prob, best_route, mode='pairwise'):
    """
    crossover
    Do not cross-mutate with the best path,
    which increases the probability of falling into the local optimal solution and requires optimization
    :param route_mat: route matrix
    :param n: number of city
    :param prob: crossover probability
    :param best_route: best route
    :return: route matrix
    """
    if mode == 'pairwise':
        route = route_mat.copy()
        np.random.shuffle(route)
        route1 = route[0::2]
        route2 = route[1::2]
        odd_r = False
        if len(route_mat) % 2 == 1:
            odd_r = True
            # odd number, ensure that route1 and route2 have the same dimension
            # route2 = np.append(route2, route1[-1][np.newaxis, ...]).reshape(route1.shape)
            # or
            route2 = np.concatenate((route2, route1[-1, np.newaxis, ...]), axis=0)

        # Prevent falling into local optimal solutions, pairwise crossing
        for i in range(len(route1[0])):
            # deep copy, to avoid changing the best route
            # best_gen = best_route.copy()
            # Partial-mapped crossover
            if prob >= np.random.rand():
                route1[i], route2[i] = inter_cross(n, route1[i], route2[i])
        if odd_r:
            route_mat = np.concatenate((route1, route2), axis=0)
        else:
            route_mat = np.concatenate((route1, route2[:-1]), axis=0)

    else:
        for i in range(len(route_mat)):
            # deep copy, to avoid changing the best route
            best_gen = best_route.copy()
            # Partial-mapped crossover
            if prob >= np.random.rand():
                route_mat[i], best_gen = inter_cross(n, route_mat[i], best_gen)

    return route_mat


def inter_cross(n, ind_a, ind_b):
    """ Partial-mapped crossover """
    r1 = np.random.randint(n)
    r2 = np.random.randint(n)
    while r2 == r1:
        r2 = np.random.randint(n)
    left, right = min(r1, r2), max(r1, r2)
    ind_a1 = ind_a.copy()
    ind_b1 = ind_b.copy()
    for i in range(left, right + 1):
        ind_a2 = ind_a.copy()
        ind_b2 = ind_b.copy()
        # Exchange
        ind_a[i] = ind_b1[i]
        ind_b[i] = ind_a1[i]
        # Each individual contains a unique serial number of the city,
        # so if the two are not the same when crossing, there will be a conflict
        x = np.argwhere(ind_a == ind_a[i])
        y = np.argwhere(ind_b == ind_b[i])
        # If a conflict occurs, replace the data
        # that is not the crossover interval with the original value to ensure that the city serial number is unique
        if len(x) == 2:
            ind_a[x[x != i]] = ind_a2[i]
        if len(y) == 2:
            ind_b[y[y != i]] = ind_b2[i]
    return ind_a, ind_b


def mutation(route_mat, n, prob=0.01):
    """
    mutation
    :param route_mat: route matrix
    :param n: number of city
    :param prob: mutation probability
    :return: route matrix
    """
    for i in range(len(route_mat)):
        if prob >= np.random.rand():
            r1 = np.random.randint(n)
            r2 = np.random.randint(n)
            while r2 == r1:
                r2 = np.random.randint(n)
            if r1 > r2:
                temp = r1
                r1 = r2
                r2 = temp
            # reverse the route
            route_mat[i, r1:r2] = route_mat[i, r1:r2][::-1]
    return route_mat


def get_route_dist(route, dist_mat):
    """calculate the distance of a route"""
    dist_sum = 0
    for i in range(len(route) - 1):
        dist_sum += dist_mat[route[i], route[i + 1]]
    dist_sum += dist_mat[route[len(route) - 1], route[0]]
    return dist_sum


def get_all_dist(route_mat, dist_mat):
    """calculate the distance of all routes"""
    dist_all = np.zeros(len(route_mat))
    for i in range(len(route_mat)):
        dist_all[i] = get_route_dist(route_mat[i], dist_mat)
    return dist_all


def main(
        coord_path,
        n_route=100,
        epoch=10000,
        dist_mode='proj',
        cross_prob=0.8,
        mut_prob=0.01,
        seed=42,
        log_dir='./log',
        filename='tsp.log',
        img_dir='./img'):
    coords = pd.read_csv(coord_path).iloc[1:61, 5]
    coords = np.array(list(list(float(x) for x in coord.split(',')) for coord in coords))
    # print(coords, type(coords))
    n = len(coords)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = cal_distance(coords[i], coords[j], dist_mode)
                dist_mat[i, j] = dist if dist != 0 else 1e-4
            else:
                dist_mat[i, j] = 1e-4
    sys.stdout = Logger(log_dir, filename=filename)

    # Init route
    route_mat = init_route(n_route, n)
    # Fitness function
    all_dist = get_all_dist(route_mat, dist_mat)
    best_route_idx = all_dist.argmin()
    # save best route and min distance
    best_route_lst = []
    min_dist_lst = []
    avg_dist_lst = []
    # The optimal path and distance are not recorded before iteration
    cur_best_route, cur_min_dist, cur_avg_dist = route_mat[best_route_idx], all_dist[best_route_idx], all_dist.mean()
    # Record the optimal route and minimum so far
    best_route, min_dist = cur_best_route, cur_min_dist
    sat_idx = 0

    start = time.time()
    for i in range(epoch):
        route_mat = selection(all_dist, route_mat)
        route_mat = crossover(route_mat, n, prob=cross_prob, best_route=best_route, mode='best')
        route_mat = mutation(route_mat, n, prob=mut_prob)
        all_dist = get_all_dist(route_mat, dist_mat)
        best_route_idx = all_dist.argmin()
        cur_best_route, cur_min_dist, cur_avg_dist = route_mat[best_route_idx], all_dist[
            best_route_idx], all_dist.mean()
        best_route_lst.append(cur_best_route)
        min_dist_lst.append(cur_min_dist)
        avg_dist_lst.append(cur_avg_dist)
        if all_dist[best_route_idx] < min_dist:
            best_route, min_dist = route_mat[best_route_idx], all_dist[best_route_idx]
            sat_idx = i + 1
        print(f'iter: {i}')
        print(f'current min_dist: {cur_min_dist}km')
        if i % 100 == 0 or i == epoch - 1:
            print(f'current best route: {cur_best_route}')
    print(f'time:{time.time() - start}s')
    print('all done!')
    print(f'saturation index: {sat_idx}')
    print(f'final min_dist: {min_dist}km')
    print(f'final best route: {best_route}')

    # draw best route and min distance for each iteration
    plt.figure(1, dpi=500)
    # plot the best route
    for i in range(len(coords) - 1):
        plt.plot([coords[best_route[i]][0], coords[best_route[i + 1]][0]],
                 [coords[best_route[i]][1], coords[best_route[i + 1]][1]], marker='o', c='deepskyblue', markersize=2)
    plt.plot([coords[best_route[0]][0], coords[best_route[-1]][0]],
             [coords[best_route[0]][1], coords[best_route[-1]][1]], marker='o', c='deepskyblue', markersize=2)
    # add text, Too many coordinates close, do not add
    # for i in range(len(coords)):
    #     plt.text(coords[best_route[i], 0], coords[best_route[i], 1], str(i + 1), fontdict={'weight': 'bold', 'size': 8})
    plt.text(coords[best_route[0], 0], coords[best_route[0], 1], 's', color='r',
             fontdict={'weight': 'bold', 'size': 8})
    plt.text(coords[best_route[-1], 0], coords[best_route[-1], 1], 'e', color='r',
             fontdict={'weight': 'bold', 'size': 8})
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(f'Forecast route(distance: {min_dist:.4f}km)')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # save image
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.savefig(os.path.join(img_dir, 'best_route.svg'), bbox_inches='tight')
    plt.show()

    plt.figure(2, dpi=500)
    plt.plot(range(1, epoch + 1), min_dist_lst, 'b', range(1, epoch + 1), avg_dist_lst, 'r')
    plt.legend(['min_dist', 'mean_dist'])
    plt.xlabel('epoch')
    plt.ylabel('dist')
    plt.title('Convergence curve')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig(os.path.join(img_dir, 'convergence_curve.svg'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--coord_path", type=str, default='./data/coords.csv', help="path of coordinate file")
    parser.add_argument("--n_route", type=int, default=180, help="number of routes")
    parser.add_argument("--epoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("--dist_mode", type=str, default='proj', help="distance mode")
    parser.add_argument("--cross_prob", type=float, default=0.8, help="probability of crossover")
    parser.add_argument("--mut_prob", type=float, default=0.2, help="probability of mutation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log_dir", type=str, default='./log', help="log directory")
    parser.add_argument("--filename", type=str, default='tsp.log', help="log filename")
    parser.add_argument("--img_dir", type=str, default='./data/img', help="image directory")
    args = parser.parse_args()
    main(**vars(args))
