import numpy as np
from gurobipy import Model, GRB
from sklearn.datasets import make_regression

# Generate sparse regression data using sklearn
m, n = 50, 60
A, y = make_regression(m, n, n_informative=10, random_state=0)
lmbd = 50.0
bigM = 1.5 * np.max(np.abs(A.T @ y))

# We want to solve
#   min (1/m) * ||y - Ax||_2^2 + lmbd * ||x||_0
# using a MIP solver (Gurobi here).

# Create the MIP problem
model = Model()

# Declare variables
x_var = model.addMVar(n, name="x", vtype="C")
z_var = model.addMVar(n, name="z", vtype="B")
r_var = y - A @ x_var

# Declare objective and constraints
model.setObjective((1/m) * (r_var @ r_var) + lmbd * sum(z_var), GRB.MINIMIZE)  
model.addConstr(x_var <= bigM * z_var)
model.addConstr(x_var >= -bigM * z_var)

# Solve the MIP problem
model.setParam("OutputFlag", 0)
model.optimize()

# Print results
print("Results")
print(" - sol. time : {:.2f} sec".format(model.Runtime))
print(" - obj. value: {:.2f}".format(model.ObjVal))
print(" - loss value: {:.2f}".format((1/m) * np.linalg.norm(y - A @ x_var.X, 2)**2))
print(" - non-zeros : {:.0f}".format(sum(z_var.X)))