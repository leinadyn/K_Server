import numpy as np
import math
from collections import deque


class Network:
    """
    A small class simulating a network with capacities and associated costs

    ...

    Attributes
    ----------
    n : int
        the number of vertices
    costs : np.ndarray
        the adjacency matrix containing costs
    capacities : np.ndarray
        the adjacency matric containing capacities
    s : int
        the vertex index of the source
    t : int
        the vertex index of the target/sink

    """

    def __init__(self, n: int, costs: np.ndarray = None, capacities: np.ndarray = None) -> None:
        self.n = n
        self.costs = costs
        self.capacities = capacities
        self.s = 0
        self.t = n-1

        assert self.costs.shape == self.capacities.shape
        assert self.n == len(self.costs[:, 0])
        assert self.costs.shape[0] == self.costs.shape[1]


class Min_Mean_Cycle_Cancel:
    """
    A class for solving max-flow-min-cost problem instances (with unit capacities) using the min-mean-cycle-reduction-algorithm

    ...

    Attributes
    ----------
    network : Network
        the network on which to find the max-flow with min-cost
    n : int
        the number of vertices
    flow : np.ndarray
        the adjacency matric containing the current flow
    s : int
        the vertex index of the source
    t : int
        the vertex index of the target/sink
    f_aug : Network
        the residual graph/network of the current flow

    Methods
    -------
    f_aug_network(self)
        updates the residual graph according to the current flow
    get_out_neighbours(self, m: np.ndarray, i: int)
        returns all out-neighbours
    get_in_neighbours(self, m: np.ndarray, i: int)
        returns all in-neighbours

    ford_fulkerson_max_flow(self)
        computes a max-flow using the Ford-Fulkerson-algorithm
    get_min_mean_cost_cycle(self, costs: np.ndarray, capacities: np.ndarray)
        returns if present the cycle with minimal negative mean cost of a strongly connected component; is based on Karp's theorem
    kosarajus_algorithm(self, adj_mat: np.ndarray)
        returns all strongly connected components using Kosarajus-algorithm

    min_mean_cycle_cancel(self)
        computes a max-flow-min-cost solution on the given network 
    """

    def __init__(self, network: Network, init_flow=0) -> None:
        self.network: Network = network
        self.n: int = network.n
        self.flow: np.ndarray = np.zeros((network.n, network.n)) + init_flow
        self.s: int = network.s
        self.t: int = network.t
        self.f_aug: Network = None

    def f_aug_network(self) -> np.ndarray:
        capacities = self.network.capacities - \
            self.flow + np.transpose(self.flow)
        costs = self.network.costs * \
            (self.network.capacities - self.flow) - \
            np.transpose(self.flow * self.network.costs)
        self.f_aug = Network(self.network.n, costs, capacities)
        return self.f_aug

    def get_out_neighbours(self, m: np.ndarray, i: int) -> np.ndarray:
        return np.array(np.nonzero(m[i, :]))

    def get_in_neighbours(self, m: np.ndarray, i: int) -> np.ndarray:
        return np.array(np.nonzero(m[:, i]))

    def __depth_first_path_search(self, adj_matrix: np.ndarray, visited: list) -> list:
        neighbours = self.get_out_neighbours(
            adj_matrix, visited[len(visited)-1])
        neighbours = neighbours[np.isin(neighbours, visited, invert=True)]

        if (not neighbours.any()) and len(visited) == 1:
            return []

        if neighbours.size == 0:
            adj_matrix[:, visited[len(visited)-1]] = 0
            return self.__depth_first_path_search(adj_matrix, visited[:-1])
        elif self.t in neighbours:
            visited.append(self.t)
            return visited
        else:
            visited.append(neighbours[0])
            return self.__depth_first_path_search(adj_matrix, visited)

    def ford_fulkerson_max_flow(self) -> np.ndarray:
        """Computes a max flow using the Ford-Fulkerson-algorithm

        Returns
        -------
        np.ndarray
            adjacency matrix containing the computed max-flow
        """

        self.flow = np.zeros((self.n, self.n))

        self.f_aug_network()

        f_aug_path = self.__depth_first_path_search(
            self.f_aug.capacities, [self.s])

        while(f_aug_path):
            indices = np.array([[f_aug_path[i], f_aug_path[i+1]]
                                for i in range(0, len(f_aug_path)-1)]).T.tolist()

            gamma = min(self.f_aug.capacities[tuple(indices)])

            flow_update = np.zeros((self.n, self.n))
            for k in range(0, len(indices[1])):
                i = indices[0][k]
                j = indices[1][k]
                if self.flow[j, i] != 0:
                    flow_update[j, i] = -gamma
                else:
                    flow_update[i, j] = gamma

            self.flow = self.flow + flow_update

            self.f_aug_network()
            f_aug_path = self.__depth_first_path_search(
                self.f_aug.capacities, [self.s])
        return self.flow

    def __get_length_k_shortestpath_to(self, costs: np.ndarray, capacities: np.ndarray) -> tuple:
        dp = np.ones((self.n+1, self.n))*(math.inf)
        paths = np.ones((self.n+1, self.n), dtype=object)*(-1)
        dp[0, 0] = 0
        paths[0, 0] = [0]

        for k in range(1, self.n+1):
            for v in range(0, self.n):
                in_neigh = self.get_in_neighbours(capacities, v).tolist()[0]
                for u in in_neigh:
                    if (dp[k-1, u] == math.inf):
                        continue

                    weight = costs[u, v]

                    if (dp[k, v] == math.inf):
                        dp[k, v] = dp[k-1, u] + weight
                        paths[k, v] = paths[k-1, u] + [v]
                    else:
                        if dp[k-1, u] + weight < dp[k, v]:
                            dp[k, v] = dp[k-1, u] + weight
                            paths[k, v] = paths[k-1, u] + [v]
        return dp, paths

    def __find_cycle(self, path: list, costs: np.ndarray, min_mean: float) -> tuple:
        visited = np.ones(len(path), dtype=object)*(-1)
        for i in range(0, len(path)):
            if visited[path[i]] == (-1):
                visited[path[i]] = [i]
            else:
                for j in visited[path[i]]:
                    mean_cost = sum([costs[path[k], path[k+1]]
                                     for k in range(j, i)])/(i-j)
                    if math.isclose(mean_cost, min_mean, abs_tol=1e-10):
                        # if mean_cost == min_mean:
                        return mean_cost, path[j:i+1]
                visited[path[i]].append(i)
        print("error: path not found")
        return []

    def get_min_mean_cost_cycle(self, costs: np.ndarray, capacities: np.ndarray) -> list:
        """Computes the cycle with with negative min-mean cost if present; Since it uses Karp's theorem it does only work on strongly connected components

        Parameters
        ----------
        costs: np.ndarray
            the adjacency matrix containing the costs of the graph
        capacities: np.ndarray
            the adjacency matrix containing the capacities of the graph (in this case it is equivalent to wether or not there is an edge since only unit capacities are used)

        Returns
        -------
        list
            containing the cycle path (the list contains the node indices in respective order)
        """

        dp, paths = self.__get_length_k_shortestpath_to(costs, capacities)
        mean = np.ones(self.n)*(-math.inf)

        for i in range(0, self.n):
            if (dp[self.n, i] == math.inf):
                continue
            for k in range(0, self.n):
                if (dp[k, i] != math.inf):
                    mean_length = (dp[self.n, i]-dp[k, i])/(self.n-k)
                    if mean_length > mean[i]:
                        mean[i] = mean_length

        mean[mean == -math.inf] = math.inf
        v = np.argmin(mean)
        min_mean = mean[v]

        if (min_mean >= 0):
            return []
        else:
            cost = min_mean
            r_path = paths[self.n, v]

        mean_cost, cycle = self.__find_cycle(r_path, costs, min_mean)
        return cycle

    def __switch_0_and_start(self, start: int, costs: np.ndarray, capacities: np.ndarray) -> tuple:
        costs_permutated = np.copy(costs)
        costs_permutated[:, [0, start]] = costs_permutated[:, [start, 0]]
        costs_permutated[[0, start], :] = costs_permutated[[start, 0], :]

        capacities_permutated = np.copy(capacities)
        capacities_permutated[:, [0, start]
                              ] = capacities_permutated[:, [start, 0]]
        capacities_permutated[[0, start],
                              :] = capacities_permutated[[start, 0], :]

        return costs_permutated, capacities_permutated

    def kosarajus_algorithm(self, adj_mat: np.ndarray) -> list:
        """Matches all vertices with their strongly connected component

        Parameters
        ----------
        adj_mat: np.ndarray
            the adjacency matrix of the graph on which to find the strongly connected components

        Returns
        -------
        list
            a list with indices of the respective strongly connected component for each vertex
        """

        visited = [-1 for k in range(0, adj_mat.shape[0])]
        L = deque()
        assigned = [-1 for k in range(0, adj_mat.shape[0])]

        for v in range(0, len(visited)):
            self.__kosarajus_visit(visited, L, adj_mat, v)
        for u in L:
            self.__kosarajus_assign(adj_mat, assigned, u, u)

        return assigned

    def __kosarajus_assign(self, adj_mat: np.ndarray, assigned: list, u: int, root: int) -> None:
        if (assigned[u] == -1):
            assigned[u] = root
            in_neighbours = self.get_in_neighbours(adj_mat, u).tolist()[0]
            for v in in_neighbours:
                self.__kosarajus_assign(adj_mat, assigned, v, root)

    def __kosarajus_visit(self, visited: list, L: list, adj_mat: np.ndarray, u: int) -> None:
        if (visited[u] == -1):
            visited[u] = True
            out_neighbours = self.get_out_neighbours(adj_mat, u).tolist()[0]
            for n in out_neighbours:
                self.__kosarajus_visit(visited, L, adj_mat, n)
            L.appendleft(u)

    def __prep_adj(self, adj_mat: np.ndarray) -> list:

        str_con_comp_list = self.kosarajus_algorithm(adj_mat)
        str_con_comp_array = np.array(str_con_comp_list)
        str_con_comp_unique = set(str_con_comp_list)

        res_str_con_com = []

        for i in str_con_comp_unique:
            cur_indices = (str_con_comp_array == i)
            other_indices = (str_con_comp_array != i)

            if sum(cur_indices) == 1:
                adj_mat[cur_indices, :] = 0
                adj_mat[:, cur_indices] = 0
            else:
                adj_mat[np.ix_(cur_indices, other_indices)] = 0
                adj_mat[np.ix_(other_indices, cur_indices)] = 0
                res_str_con_com.append(i)

        return res_str_con_com

    def min_mean_cycle_cancel(self) -> np.ndarray:
        """Computes a max-flow with min cost on the given network

        Returns
        -------
        np.ndarray
            the adjacency matrix containing the computed max-flow- with min-cost 
        """

        # find a max-flow
        self.ford_fulkerson_max_flow()

        # reduce costs of max-flow if possible
        while(True):
            self.f_aug_network()

            start_ind = self.__prep_adj(self.f_aug.capacities)

            if len(start_ind) == 0:
                break

            unsuccessful_counter = 0

            for start in start_ind:
                costs_permutated, capacities_permutated = self.__switch_0_and_start(
                    start, self.f_aug.costs, self.f_aug.capacities)
                r_path = self.get_min_mean_cost_cycle(
                    costs_permutated, capacities_permutated)
                r_path = [0 if r_path[i] == start else (
                    start if r_path[i] == 0 else r_path[i]) for i in range(0, len(r_path))]

                if not(r_path):
                    unsuccessful_counter = unsuccessful_counter+1
                    continue
                else:
                    indices = np.array([[r_path[i], r_path[i+1]]
                                        for i in range(0, len(r_path)-1)]).T.tolist()

                    gamma = min(self.f_aug.capacities[tuple(indices)])

                    flow_update = np.zeros((self.n, self.n))
                    for k in range(0, len(indices[1])):
                        i = indices[0][k]
                        j = indices[1][k]
                        if self.flow[j, i] != 0:
                            flow_update[j, i] = -gamma
                        else:
                            flow_update[i, j] = gamma

                    self.flow = self.flow + flow_update

            if unsuccessful_counter == len(start_ind):
                break

        return self.flow
