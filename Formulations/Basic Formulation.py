# Basic Model

from gurobipy import Model, GRB, quicksum
import numpy as np
import time

T = 168  # Time horizon
L = 8  # Minimum-up time
ℓ = 8  # Minimum-down time
C_max = 455  # Maximum generation (MW)
C_min = 150  # Minimum generation (MW)
V_up = 91  # Ramp-up (MW)
V_start = 180  # Startup (MW)
SU = 2000  # Start-up cost
SD = 1000  # Shut-down cost
a, b, c = 0.00048, 16.19, 1000  # Quadratic cost coefficients

# Electricity prices for each time period (example)
np.random.seed(42)  # For reproducibility
price = np.random.uniform(0, 35, T)


def create_model(relax=False):
    # Create a model
    model = Model("SelfSchedulingUC")

    # Decision variables
    x = model.addVars(T, name="x")  # Power generation
    y = model.addVars(T, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="y")  # On/Off status
    u = model.addVars(T, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="u")  # Start-up decision

    # Objective function
    model.setObjective(quicksum(price[t] * x[t] - (a * x[t] * x[t] + b * x[t] + c * y[t]) for t in range(T)) -
                       quicksum(SU * u[t] + SD * (y[t - 1] - y[t] + u[t]) for t in range(1, T)),
                       GRB.MAXIMIZE)

    # Constraints (1a)
    for t in range(L, T):
        model.addConstr(quicksum(u[i] for i in range(t - L + 1, t + 1)) <= y[t], f"MinUpTime_{t}")

    # Constraints (1b)
    for t in range(ℓ, T):
        model.addConstr(quicksum(u[i] for i in range(t - ℓ + 1, t + 1)) <= 1 - y[t - ℓ], f"MinDownTime_{t}")

    # Constraints (1c)
    for t in range(1, T):
        model.addConstr(y[t] - y[t - 1] - u[t] <= 0, f"LogicConstraint_{t}")

    # Constraints (1d)
    for t in range(T):
        model.addConstr(x[t] >= C_min * y[t], f"GenLowerBound_{t}")

    # Constraints (1e)
    for t in range(T):
        model.addConstr(x[t] <= C_max * y[t], f"GenUpperBound_{t}")

    # Constraints (1f)
    for t in range(1, T):
        model.addConstr(x[t] - x[t - 1] <= V_up * y[t - 1] + V_start * (1 - y[t - 1]), f"RampUp_{t}")

    # Constraints (1g)
    for t in range(1, T):
        model.addConstr(x[t - 1] - x[t] <= V_up * y[t] + V_start * (1 - y[t]), f"RampDown_{t}")

    return model, x, y, u


start = time.time()
# Solve MILP model
milp_model, x_milp, y_milp, u_milp = create_model(relax=False)
milp_model.optimize()
end = time.time()
total_time = end - start
Z_MILP = milp_model.ObjVal

# Solve LP relaxation model
lp_model, x_lp, y_lp, u_lp = create_model(relax=True)
lp_model.optimize()
Z_LP = lp_model.ObjVal

# Calculate integrality gap
integrality_gap = (Z_LP - Z_MILP) / Z_LP * 100

# Display results
print(f"\nOptimal MILP Objective Value: {Z_MILP:.2f}")
print(f"Optimal LP Relaxation Objective Value: {Z_LP:.2f}")
print(f"Integrality Gap: {integrality_gap:.2f}%")
print(f"Time to solve: {total_time}")
# Display MILP solution
print("\nMILP Solution:")
for t in range(T):
    print(f"Time {t}: x = {x_milp[t].X:.2f}, y = {y_milp[t].X}, u = {u_milp[t].X}")
