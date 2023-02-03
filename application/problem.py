import source.Solver as ks
import math
import random

class Point_2D:
    """
    A small class simulating a location of a server or request on the plane

    ...

    Attributes
    ----------
    x : int
        x position
    x : int
        y position

    Methods
    -------
    dist(other)
        computes the distance between this location and another one 
    
    """
    
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        
    def __str__(self) -> str:
        return f'({self.x},{self.y})'
    
    def __repr__(self) -> str:
        return f'({self.x},{self.y})'
        
    def dist(self, other) -> float:
        return abs(math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2))


problem = None
requests_config = []
initial_servers = [Point_2D(50,10), Point_2D(70,10), Point_2D(90,10), Point_2D(110,10), Point_2D(130,10)]
solver = []


def set_request_config(k):
    global requests_config
    if k==0:
        requests_config = [Point_2D(350, 490), Point_2D(280, 470), Point_2D(360, 370), Point_2D(220, 365), Point_2D(360, 370), Point_2D(190, 410), Point_2D(230, 450), Point_2D(360, 370), 
        Point_2D(240, 500), Point_2D(360, 370)]
        requests_config_js = [[350,490], [280,470], [360,370], [220,365], [360,370], [190,410], [230,450], [360,370], [240,500], [360,370]]
    elif k==1:
        requests_config = [Point_2D(110,150), Point_2D(350,260), Point_2D(200,240), Point_2D(300,120), Point_2D(140,400), Point_2D(100,170), Point_2D(400,400), Point_2D(245,278),
        Point_2D(178,150), Point_2D(335,155)]
        requests_config_js = [[110,150], [350,260], [200,240], [300,120], [140,400], [100,170], [400,400], [245,278], [178,150], [335,155]]
    elif k==2:
        requests_config.clear()
        requests_config_js = []
        for k in range(0,10):
            x = random.choice(range(110,400))
            y = random.choice(range(70,500))
            requests_config.append(Point_2D(x,y))
            requests_config_js.append([x,y])
    elif k==3:
        requests_config = [Point_2D(340,65), Point_2D(370,120), Point_2D(330,220), Point_2D(405,260), Point_2D(380,480), Point_2D(230,525), Point_2D(140,420), Point_2D(110,295), 
        Point_2D(120,160), Point_2D(150,70)]
        requests_config_js = [[340,65], [370,120], [330,220], [405,260], [380,480], [230,525], [140,420], [110,295], [120,160], [150,70]]
    else:
        requests_config.clear()
        requests_config_js = []
        spot_centers = [[140,250],[370,400],[300,130]]
        for i in range(0,10):
            spot = random.choice(range(0,3))
            deviate_x = random.choice(range(1,4))
            deviate_y = random.choice(range(1,4))
            x = spot_centers[spot][0]+5*deviate_x
            y = spot_centers[spot][1]+5*deviate_y
            requests_config.append(Point_2D(x,y))
            requests_config_js.append([x,y])
    assert len(requests_config)==len(requests_config_js)
    return requests_config_js



def set_problem(num_server, pred_solver, pred_error):
    global requests_config, initial_servers, solver, problem
    problem = ks.K_Server(initial_servers[:num_server], requests_config)

    solver = [problem.opt_solver, problem.random_solver, problem.greedy_solver, problem.wfa_network_solver, problem.ftp_solver]
    assert pred_solver != 4
    problem.generate_prediction_list(solver[pred_solver], pred_error)


def solve_problem():
    global solver, problem
    opt_net = problem.execute_opt_network()[0]
    random = problem.execute_random()[0]
    greedy = problem.execute_greedy()[0]
    ftp = problem.execute_ftp()[0]
    wfa = problem.execute_wfa()[0]

    eps = 0.01

    det_comb_of_ftp_opt = problem.execute_combine_deterministic([solver[4], solver[0]])
    ran_comb_of_greedy_wfa = problem.execute_combine_randomized([solver[4], solver[0]], eps)

    return [opt_net, random, greedy, wfa, ftp, det_comb_of_ftp_opt, ran_comb_of_greedy_wfa]


def get_cost(index_list):
    global problem
    return problem.get_cost(index_list)
