import numpy as np
import math
from source import Network as net


class K_Server:
    """
    A class simulating a K-Server-Problem instance and several offline and online solvers

    ...

    Attributes
    ----------
    servers : list
        a list containing the location objects of all servers
    requests : list
        a list containing the location objects of all requests
    pos : list
        concatenated server and request locations
    diameter : float, optional
        the metric space diameter, if not stated it will be computed
    prediction_list : list, optional
        a list with predicted configurations for each request; can be stated or generated with a solver

    Methods
    -------
    opt_solver(s_before, cur_request)
        An optimal solver returning an index of the server to be moved to the current request 
    random_solver(s_before, cur_request)
        A solver for random assignment returning an index of the server to be moved to the current request
    greedy_solver(s_before, cur_request)
        A solver for the greedy algorithm returning an index of the server to be moved to the current request
    ftp_solver(s_before, cur_request)
        A solver for FollowThePrediction returning an index of the server to be moved to the current request
    wfa_network_solver(s_before, cur_request)
        A solver for the work function algorithm returning an index of the server to be moved to the current request

    execute_opt_dp()
        Computes an optimal offline solution of the problem instance via dynamic programming
    execute_opt_network()
        Computes an optimal offline solution of the problem instance by solving a max-flow-min-cost problem instance
    execute_random()
        Computes an online solution of the problem instance using random server assignment
    execute_greedy()
        Computes an online solution of the problem instance using a greedy heuristic
    execute_ftp()
        Computes an online solution of the problem instance via the FollowThePrediction-algorithm
    execute_wfa()
        Computes an online solution of the problem instance via work function algorithm
    execute_combine_deterministic(solver)
        Computes an online solution of the problem instance by deterministically combining different solvers 
    execute_combine_randomized(solver, eps)
        Computes an online solution of the problem instance by randomly combining different solvers

    generate_prediction_list(solver, error_prob=0)
        generates predicted configurations for ftp using the given solver and deviation probability
    """

    def __init__(self, init_config: list, requests: list, prediction_list: list = None, diameter: float = None):
        self.servers: list = init_config
        self.requests: list = requests
        self.pos: list = init_config + requests
        self.diameter: float = diameter
        if self.diameter == None:
            self.diameter = self.__get_max_dist()
        self.prediction_list: list = prediction_list

    def execute_opt_dp(self) -> tuple:
        """Computes an optimal offline solution by solving it as a dynamic program

        Returns
        -------
        tuple
            a tuple containing the total cost of the used configurations and a list with an index of the assigned server for each request 

        Notes
        -------
        This approach can be really slow for many requests or servers. It can however sometimes find a different optimal solution 
        than the network approach which can make it interesting for small problems.
        """
        cur_request_index = len(self.servers)
        config = range(0, len(self.servers))
        return self.__opt_dp_min_search(config, cur_request_index)

    def __opt_dp_min_search(self, config: list, cur_request: int) -> tuple:
        cur_best = math.inf

        for i in range(0, len(config)):
            cur_cost = self.pos[config[i]].dist(self.pos[cur_request])
            if len(self.pos)-1 == (cur_request):
                cost = cur_cost
                further_indices = []
            else:
                further_costs, further_indices = self.__opt_dp_min_search([config[k] if k != i else cur_request
                                                                           for k in range(0, len(config))], cur_request+1)
                cost = cur_cost + further_costs

            if cost < cur_best:
                cur_best = cost
                cur_index_list = [i] + further_indices

        return cur_best, cur_index_list

    def __get_max_dist(self) -> float:
        if self.diameter != None:
            return self.diameter
        else:
            best_dist = 0
            for i in range(0, len(self.pos)):
                for j in range(0, len(self.pos)):
                    if i != j:
                        cur_dist = self.pos[i].dist(self.pos[j])
                        if cur_dist > best_dist:
                            best_dist = cur_dist
            return best_dist

    def __opt_network(self) -> tuple:
        num_serv = len(self.servers)
        num_req = len(self.requests)
        n_nodes = 1 + num_serv + 2*num_req + 1
        # s, s_i, r_i, r'_i, t
        capacities = np.zeros((n_nodes, n_nodes))
        costs = np.zeros((n_nodes, n_nodes))

        capacities[0, range(1, num_serv+1)] = 1

        for i in range(0, num_req):
            capacities[num_serv+1+i, num_serv+1+num_req+i] = 1
            costs[num_serv+1+i, num_serv+1+num_req+i] = - \
                (2*self.diameter + 1)  # needs to be > 2*max_dist

        for i in range(0, num_serv):
            capacities[i+1, n_nodes-1] = 1

            capacities[i+1, range(num_serv+1, num_serv+1+num_req)] = 1
            for j in range(0, num_req):
                costs[i+1, 1+num_serv +
                      j] = self.servers[i].dist(self.requests[j])

        for i in range(0, num_req):
            capacities[i+num_serv+num_req+1, n_nodes-1] = 1

            for j in range(i+1, num_req):
                capacities[i+num_serv+num_req+1, 1+num_serv+j] = 1
                costs[i+num_serv+num_req+1, 1+num_serv +
                      j] = self.requests[i].dist(self.requests[j])

        return capacities, costs

    def opt_solver(self, s_before: list, cur_request: int) -> int:
        """Computes an index of the server to be moved to the current request following the optimal strategy

        Parameters
        ----------
        s_before : list
            the server configuration (a list with a position index for each server referring to self.pos) from the last iteration
        cur_request : int
            the index of the current request; is equivalent to the current time when executing the online algorithm

        Returns
        -------
        int
            the index of the server to be moved to the current request

        Notes
        -------
        Simulates a solver by computing the optimal offline solution to be used when generating predictions for ftp. 
        """
        cost, index_list = self.execute_opt_network()
        return index_list[cur_request]

    def execute_opt_network(self) -> tuple:
        """Computes an optimal offline solution by transforming the problem to a max-flow-min-cost problem and solving it

        Returns
        -------
        tuple
            a tuple containing the total cost of the used configurations and a list with an index of the assigned server for each request 

        Notes
        -------
        This approach is much faster than the other recursive one and to be preffered when having many requests and/or servers.
        """
        capacities, costs = self.__opt_network()

        n_nodes = costs.shape[0]
        network = net.Network(n_nodes, costs, capacities)
        optimizer = net.Min_Mean_Cycle_Cancel(network)

        opt_flow = optimizer.min_mean_cycle_cancel()

        index_list = [-1 for k in range(0, len(self.requests))]
        for i in range(0, len(self.servers)):
            req = self.__opt_network_find_path(opt_flow, i+1, [])
            for k in req:
                index_list[k] = i
        cost = self.get_cost(index_list)
        return cost, index_list

    def get_cost(self, index_list: list) -> float:
        config = list(range(0, len(self.servers)))
        cost = 0
        
        for r in range(0, len(index_list)):
            cost = cost + self.pos[config[index_list[r]]
                                   ].dist(self.requests[r])
            config[index_list[r]] = len(self.servers) + r
        return cost

    def __opt_network_find_path(self, flow: np.ndarray, i: int, req: list) -> list:
        index = np.where(flow[i, :] != 0)[0][0]
        # 1 + len(self.servers) + 2*len(self.requests):
        if index == (flow.shape[0]-1):
            return req
        if index >= 1+len(self.servers) and index < 1+len(self.servers)+len(self.requests):
            req.append(index-1-len(self.servers))
        return self.__opt_network_find_path(flow, index, req)

    def __wfa_network(self, s_before: list, r: int) -> tuple:
        num_serv = len(self.servers)
        num_req = r+1
        n_nodes = 1 + 2*num_serv + 2*num_req + 1
        # s, s_i, r_i, s'_i, r'_i, t
        capacities = np.zeros((n_nodes, n_nodes))
        costs = np.zeros((n_nodes, n_nodes))

        capacities[0, range(1, num_serv+1)] = 1

        for i in range(0, num_req):
            capacities[num_serv+1+i, 2*num_serv+1+num_req+i] = 1
            costs[num_serv+1+i, 2*num_serv+1 +
                  num_req+i] = -(2*self.diameter + 1)

        for i in range(0, num_serv):
            capacities[i+1, range(num_serv+1, num_serv+1+num_req)] = 1
            for j in range(0, num_req):
                costs[i+1, 1+num_serv +
                      j] = self.servers[i].dist(self.requests[j])

            capacities[i+1, range(num_serv+1+num_req, 2 *
                                  num_serv+1+num_req)] = 1
            for j in range(0, num_serv):
                costs[i+1, 1+num_serv+num_req +
                      j] = self.servers[i].dist(self.pos[s_before[j]])

        for i in range(0, num_req-1):
            capacities[i+1+2*num_serv+num_req,
                       range(num_serv+1+num_req, 2*num_serv+1+num_req)] = 1
            for j in range(i+1, num_serv):
                costs[i+1+2*num_serv+num_req, 1+num_serv+num_req +
                      j] = self.servers[i].dist(self.pos[s_before[j]])

            for j in range(i+1, num_req):
                capacities[i+2*num_serv+num_req+1, 1+num_serv+j] = 1
                costs[i+2*num_serv+num_req+1, 1+num_serv +
                      j] = self.requests[i].dist(self.requests[j])

        capacities[2*num_serv+2*num_req, n_nodes-1] = 1

        x = 1/(num_serv-1)*sum([self.requests[num_req -
                                              1].dist(self.pos[s_before[j]]) for j in range(0, num_serv)])

        for i in range(0, num_serv):
            capacities[i+num_serv+num_req+1, n_nodes-1] = 1
            costs[i+num_serv+num_req+1, n_nodes-1] = x - \
                self.requests[num_req-1].dist(self.pos[s_before[i]])

        return capacities, costs

    def wfa_network_solver(self, s_before: list, cur_request: int) -> int:
        """Computes an index of the server to be moved to the current request using the work function algorithm

        Parameters
        ----------
        s_before : list
            the server configuration (a list with a position index for each server referring to self.pos) from the last iteration
        cur_request : int
            the index of the current request; is equivalent to the current time when executing the online algorithm

        Returns
        -------
        int
            the index of the server to be moved to the current request

        Notes
        -------
        The solution of the work function algorithm is computed by transforming the problem to a max-flow-min-cost problem
        and solving it using the min-mean-cycle-reduction algorithm.
        See Tomislav Rudec & Alfonzo Baumgartner & Robert Manger, 2013. "A fast work function algorithm for solving the k-server problem," for details.
        """

        capacities, costs = self.__wfa_network(s_before, cur_request)

        n_nodes = costs.shape[0]
        network = net.Network(n_nodes, costs, capacities)
        optimizer = net.Min_Mean_Cycle_Cancel(network)

        opt_flow = optimizer.min_mean_cycle_cancel()

        index_list = [-1 for k in range(0, cur_request+1)]
        for i in range(0, len(self.servers)):
            req = self.__wfa_network_find_path(opt_flow, i+1, [], cur_request)
            for k in req:
                index_list[k] = i
        return index_list[cur_request]

    def __wfa_network_find_path(self, flow: np.ndarray, i: int, req: list, cur_request: int) -> list:
        index = np.where(flow[i, :] != 0)[0][0]
        # 1 + len(self.servers) + 2*len(self.requests):
        if index == (flow.shape[0]-1):
            return req
        if index >= 1+len(self.servers) and index <= 1+len(self.servers)+cur_request:
            req.append(index-1-len(self.servers))
        return self.__wfa_network_find_path(flow, index, req, cur_request)

    def greedy_solver(self, s_before: list, cur_request: int) -> int:
        """Computes an index of the server to be moved to the current request using a greedy approach, i.e. always choosing the closest server

        Parameters
        ----------
        s_before : list
            the server configuration (a list with a position index for each server referring to self.pos) from the last iteration
        cur_request : int
            the index of the current request; is equivalent to the current time when executing the online algorithm

        Returns
        -------
        int
            the index of the server to be moved to the current request
        """

        cur_best = self.pos[s_before[0]].dist(self.requests[cur_request])
        cur_index = 0
        for i in range(1, len(self.servers)):
            dist = self.pos[s_before[i]].dist(self.requests[cur_request])
            if dist < cur_best:
                cur_index = i
                cur_best = dist
        return cur_index

    def random_solver(self, s_before: list, cur_request: int) -> int:
        """Computes an index of the server to be moved to the current request by random assignment; Each server has the same probability

        Parameters
        ----------
        s_before : list
            the server configuration (a list with a position index for each server referring to self.pos) from the last iteration
        cur_request : int
            the index of the current request; is equivalent to the current time when executing the online algorithm

        Returns
        -------
        int
            the index of the server to be moved to the current request
        """

        return np.random.choice(range(0, len(self.servers)))

    def ftp_solver(self, s_before: list, cur_request: int) -> int:
        """Computes an index of the server to be moved to the current request using the FollowThePrediction approach; does require a prediction_list to be generated or stated; see https://arxiv.org/abs/2003.02144 for details

        Parameters
        ----------
        s_before : list
            the server configuration (a list with a position index for each server referring to self.pos) from the last iteration
        cur_request : int
            the index of the current request; is equivalent to the current time when executing the online algorithm

        Returns
        -------
        int
            the index of the server to be moved to the current request

        Notes
        -------
        Note that in this implementation it will always move the server which results in a configuration closest to the prediction.
        Since the cost for a task is either 0 if served and infinity if not, it does only allow configs which serve a request.
        Also it does not makes sense to move more than just one server, which could normally happen in ftp, since it just does inflict more costs than neccessary. 
        """

        prediction = self.prediction_list[cur_request]
        cur_best = math.inf
        cur_request_index = len(self.servers) + cur_request
        index = -1

        for i in range(0, len(s_before)):
            # do i have to check every possible config? -> does not make sense for k-server
            # configs without serving request can be ignored -> cost inf
            x = [s_before[k] if k !=
                 i else cur_request_index for k in range(0, len(s_before))]
            config_dist = sum([self.pos[x[j]].dist(
                self.pos[prediction[j]]) for j in range(0, len(x))])

            if 2*config_dist < cur_best:
                cur_best = 2*config_dist
                index = i
        return index

    def execute_random(self) -> tuple:
        """Computes an online solution by using the random assignment solver

        Returns
        -------
        tuple
            a tuple containing the total cost of the used configurations and a list with an index of the assigned server for each request 
        """

        num_serv = len(self.servers)
        num_req = len(self.requests)
        req_serv = []

        config = range(0, num_serv)
        cost = 0

        for time in range(0, num_req):
            move = self.random_solver(config, time)
            req_serv.append(move)
            cost = cost + self.pos[config[move]].dist(self.pos[num_serv+time])
            config = [config[i] if i != move else num_serv +
                      time for i in range(0, num_serv)]
        return cost, req_serv

    def execute_greedy(self) -> tuple:
        """Computes an online solution by using the greedy solver

        Returns
        -------
        tuple
            a tuple containing the total cost of the used configurations and a list with an index of the assigned server for each request 
        """

        num_serv = len(self.servers)
        num_req = len(self.requests)
        req_serv = []

        config = range(0, num_serv)
        cost = 0

        for time in range(0, num_req):
            move = self.greedy_solver(config, time)
            req_serv.append(move)
            cost = cost + self.pos[config[move]].dist(self.pos[num_serv+time])
            config = [config[i] if i != move else num_serv +
                      time for i in range(0, num_serv)]
        return cost, req_serv

    def execute_ftp(self) -> tuple:
        """Computes an online solution by using the FollowThePrediction algorithm solver; does require a prediction_list to be generated or stated; see https://arxiv.org/abs/2003.02144 for details

        Returns
        -------
        tuple
            a tuple containing the total cost of the used configurations and a list with an index of the assigned server for each request 
        """

        assert self.prediction_list != None
        num_serv = len(self.servers)
        num_req = len(self.requests)
        req_serv = []

        config = range(0, num_serv)
        cost = 0

        for time in range(0, num_req):
            move = self.ftp_solver(config, time)
            req_serv.append(move)
            cost = cost + self.pos[config[move]].dist(self.pos[num_serv+time])
            config = [config[i] if i != move else num_serv +
                      time for i in range(0, num_serv)]
        return cost, req_serv

    def execute_wfa(self) -> tuple:
        """Computes an online solution by using the work function algorithm solver

        Returns
        -------
        tuple
            a tuple containing the total cost of the used configurations and a list with an index of the assigned server for each request 
        """

        num_serv = len(self.servers)
        num_req = len(self.requests)
        req_serv = []

        config = range(0, num_serv)
        cost = 0

        for time in range(0, num_req):
            move = self.wfa_network_solver(config, time)
            req_serv.append(move)
            cost = cost + self.pos[config[move]].dist(self.pos[num_serv+time])
            config = [config[i] if i != move else num_serv +
                      time for i in range(0, num_serv)]
        return cost, req_serv

    def execute_combine_deterministic(self, solver: list) -> float:
        """Computes an online solution by combining multiple solvers in a deterministic way; see https://arxiv.org/abs/2003.02144 for details

        Parameters
        ----------
        solver : list[method]
            the list of solvers to be combined; can be two or arbitrary many solvers

        Returns
        -------
        float
            a tuple containing the total cost of the used configurations
        """

        num_serv = len(self.servers)
        num_req = len(self.requests)

        l = 0
        m = len(solver)
        gamma = m/(m-1)
        configs = [list(range(0, len(self.servers))) for s in range(0, m)]
        cur_costs = [0 for s in range(0, m)]

        time = 0
        cur_config = configs[0]
        cur_cost = 0

        i = l % m
        while(True):
            for j in range(0, m):
                move = solver[j](configs[j], time)
                cur_costs[j] = cur_costs[j] + self.pos[configs[j]
                                                       [move]].dist(self.pos[num_serv+time])
                configs[j] = [configs[j][i] if i !=
                              move else num_serv+time for i in range(0, num_serv)]
            while (cur_costs[i] > gamma**l):
                l = l+1
                i = l % m
            cost_dif = sum([self.pos[cur_config[j]].dist(
                self.pos[configs[i][j]]) for j in range(0, num_serv)])
            cur_cost = cur_cost + cost_dif
            cur_config = configs[i]
            time = time + 1
            if (time == num_req):
                break
        return cur_cost

    def execute_combine_randomized(self, solver: list, eps: float) -> float:
        """Computes an online solution by combining two solvers in a randomized way; see https://arxiv.org/abs/2003.02144 for details

        Parameters
        ----------
        solver : list[method]
            the list of solvers to be combined; must have length=2
        eps : float
            a positive number parameter of the algorithm; must be smaller than 0.5

        Returns
        -------
        float
            a tuple containing the total cost of the used configurations
        """

        num_serv = len(self.servers)
        num_req = len(self.requests)

        m = len(solver)
        assert m == 2
        assert eps < .5 and eps > 0

        beta = 1 - eps/2
        w = [1 for i in range(0, m)]

        configs = [list(range(0, len(self.servers))) for s in range(0, m)]
        cur_costs = [0 for s in range(0, m)]

        time = 0
        cur_config = configs[0]
        cur_cost = 0

        i = 0
        select_probs = [1, 0]
        p_new = [1, 0]

        while(True):
            for j in range(0, m):
                move = solver[j](configs[j], time)
                cur_costs[j] = cur_costs[j] + self.pos[configs[j]
                                                       [move]].dist(self.pos[num_serv+time])
                configs[j] = [configs[j][i] if i !=
                              move else num_serv+time for i in range(0, num_serv)]
            w = [w[i]*(beta ** (cur_costs[i]/self.diameter))
                 for i in range(0, m)]
            w_sum = sum(w)
            p_old = p_new.copy()
            p_new = [w[i]/w_sum for i in range(0, m)]

            j = (i+1) % m
            tau = max(0, p_old[j]-p_new[j])
            select_probs[j] = tau/p_old[i]
            select_probs[i] = 1-select_probs[j]
            i = np.random.choice([0, 1], p=select_probs)

            cost_dif = sum([self.pos[cur_config[j]].dist(
                self.pos[configs[i][j]]) for j in range(0, num_serv)])
            cur_cost = cur_cost + cost_dif
            cur_config = configs[i]

            time = time + 1
            if (time == num_req):
                break
        return cur_cost

    def generate_prediction_list(self, solver, error_prob: float = 0) -> tuple:
        """Generates a list with predicted configurations which can be used by the ftp_solver

        Parameters
        ----------
        solver : method
            the solver used to generate the configurations for each request
        error_prob : float, optional
            probability with which the predicted configuration will deviate from the one suggested by the solver 
            (default is 0)

        Returns
        -------
        tuple
            a tuple of 
                1. predicted configurations (a configuration is a list with a position index for each server referring to self.pos)
                2. a list with an assigned server for every request; only used for the app
        """

        assert error_prob >= 0 and error_prob <= 1

        num_serv = len(self.servers)
        num_req = len(self.requests)
        prediction_list = []

        config = range(0, num_serv)
        assignment_list = []

        for time in range(0, num_req):
            move = solver(config, time)
            move = np.random.choice(
                [move, (move+1) % num_serv], p=[1-error_prob, error_prob])
            config = [config[i] if i != move else num_serv +
                      time for i in range(0, num_serv)]
            prediction_list.append(config)
            print(move)
            assignment_list.append(move)
        self.prediction_list = prediction_list
        return prediction_list, assignment_list
