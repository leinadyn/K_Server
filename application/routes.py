from application import app
from flask import render_template, url_for, request
import application.problem as p


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_requests")
def get_requests_and_initialize():
    k = int(request.args.get('k'))
    num_server = int(request.args.get('num_server'))
    pred_solver = int(request.args.get('pred_solver'))
    pred_error = float(request.args.get('pred_error'))
    requests = p.set_request_config(k)
    p.set_problem(num_server, pred_solver, pred_error)
    return requests

@app.route("/solve")
def solve():
    index_list = request.args.get('index_list')
    index_list = [int(x) for x in index_list.split(',')]
    values = p.solve_problem()
    cost = p.get_cost(index_list)
    costs = [cost]+values
    ranking = [sorted(costs).index(x) for x in costs]
    return costs + ranking


