from application import app
from flask import render_template, url_for, request
import application.problem as p


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/getRequests")
def get_requests():
    k = int(request.args.get('k'))
    requests = p.set_request_config(k)
    return requests

@app.route("/init")
def initialize_and_get_preds():
    num_server = int(request.args.get('num_server'))
    pred_solver = int(request.args.get('pred_solver'))
    deviation_prob = float(request.args.get('deviation_prob'))
    predictions = p.set_problem(num_server, pred_solver, deviation_prob)
    predictions = [int(p) for p in predictions]
    print(predictions)
    return predictions

@app.route("/solve")
def solve():
    index_list = request.args.get('index_list')
    index_list = [int(x) for x in index_list.split(',')]
    values = p.solve_problem()
    cost = p.get_cost(index_list)
    costs = [cost]+values
    ranking = [sorted(costs).index(x) + 1 for x in costs]
    order = [i[0] for i in sorted(enumerate(costs), key=lambda x:x[1])]
    row_ind = [order.index(x) for x in range(0,len(costs))]
    return costs + ranking + row_ind
