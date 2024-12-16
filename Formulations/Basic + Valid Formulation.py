# Basic Formulation with added valid inequalities

from gurobipy import Model, GRB, quicksum
import numpy as np
import time

T = 2000  # Time horizon
L = 5  # Minimum-up time
ℓ = 5  # Minimum-down time
C_max = 130  # Maximum generation (MW)
C_min = 20  # Minimum generation (MW)
V_up = 26  # Ramp-up (MW)
V_start = 35  # Startup (MW)
SU = 500  # Start-up cost
SD = 250  # Shut-down cost
a, b, c = 0.00211, 16.5, 680  # Quadratic cost coefficients

np.random.seed(42)
price = np.random.uniform(0, 44, T)


def create_model(relax=False):
    # Create a model
    model = Model("SelfSchedulingUC")

    # Decision variables
    x = model.addVars(T, name="x")
    y = model.addVars(T, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="y")
    u = model.addVars(T, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="u")

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

    for t in range(T - 1):
        # Constraint (2a)
        model.addConstr(u[t + 1] >= 0, f"Constraint_2a_1_{t}")
        model.addConstr(u[t + 1] >= y[t + 1] - y[t], f"Constraint_2a_2_{t}")

        # Constraint (2b)
        model.addConstr(u[t + 1] <= y[t + 1], f"Constraint_2b_1_{t}")
        model.addConstr(y[t] + u[t + 1] <= 1, f"Constraint_2b_2_{t}")

        # Constraint (2c)
        model.addConstr(x[t] >= C_min * y[t], f"Constraint_2c_1_{t}")
        model.addConstr(x[t + 1] >= C_min * y[t + 1], f"Constraint_2c_2_{t}")

        # Constraint (2d)
        model.addConstr(x[t] <= V_start * y[t] + (C_max - V_start) * (y[t + 1] - u[t + 1]), f"Constraint_2d_{t}")

        # Constraint (2e)
        model.addConstr(x[t + 1] <= C_max * y[t + 1] - (C_max - V_start) * u[t + 1], f"Constraint_2e_{t}")

        # Constraint (2f)
        model.addConstr(
            x[t + 1] - x[t] <= (C_min + V_up) * y[t + 1] - C_min * y[t] - (C_min + V_up - V_start) * u[t + 1],
            f"Constraint_2f_{t}")

        # Constraint (2g)
        model.addConstr(
            x[t] - x[t + 1] <= V_start * y[t] - (V_start - V_up) * y[t + 1] - (C_min + V_up - V_start) * u[t + 1],
            f"Constraint_2g_{t}")

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


