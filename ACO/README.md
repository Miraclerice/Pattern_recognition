### 蚁群算法

&emsp;&emsp;&emsp;[Ant Colony Optimization(ACO)](https://zh.wikipedia.org/wiki/%E8%9A%81%E7%BE%A4%E7%AE%97%E6%B3%95)，是一种用来在图中寻找优化路径的机率型算法。它由Marco Dorigo于1992年在他的博士论文“Ant system: optimization by a colony of cooperating agents”中提出，其灵感来源于蚂蚁在寻找食物过程中发现路径的行为。

### 问题描述

&emsp;&emsp;在一个$50 \times 50$​ 的网格中，每个格子要么是障碍物，要么是可通行的。玩家需要从指定的起点移动到终点。每次只能在上、下、左、右四个方向上移动一个格子，并且不能通过障碍物。使用蚁群算法求解从起点到终点的最短路径。
$$
p_{ij}^{k}\left(t\right)=\begin{cases}\frac{\left[\tau_{ij}\left(t\right)\right]^\alpha}{\sum_{s\in J_k\left(i\right)}\left[\tau_{is}\left(t\right)\right]^\alpha},&\quad{j\in J_k\left(i\right)}\\0,&\quad else\end{cases}
$$