# K_Server

========

This project provides several offline and online solver for the K-Server-Problem.

## Requirements

---

The project itself only requires numpy apart from base python. If you want to take a look at the jupyter notebook you also need Jupyter Notebook.

## Usage

---

A thorough explanation on how to use it and which solvers are availlable can be found in the "examples" jupyter notebook.

    import K_Server as ks

    servers = ...
    requests = ...

    problem = ks.K_Server(servers, requests)
    problem.execute_opt_dp()

## Features

---

K_Server.py contains the K_Server class, which implements the following offline solvers:

- one via dynamic programming
- one via max-flow-min-cost

And the following online solvers:

- randomized assignment
- greedy approach
- Follow the Prediction
- Work-Function-Algorithm via max-flow-min-cost
- deterministic combination of multiple algorithms
- randomized combination of two algorithms

In addition Network.py provides the following two classes (though they are only helpers for the solvers):

- a Network class, which is used as a data structure to compute network flows
- a Min_Mean_Cycle_Cancel class, which implements the min-mean-cycle-reduction algorithm for computing max-flows with minimal cost and other graph algorithms

## Contribute

---

If you should have questions or find a bug feel free to contact me.

- Issue Tracker: https://github.com/leinadyn/K_Server/issues
- Source Code: https://github.com/leinadyn/K_Server

## Support

---

If you are having issues, please let me know.
