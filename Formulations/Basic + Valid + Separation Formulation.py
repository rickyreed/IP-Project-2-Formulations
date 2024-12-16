# Separation with valid inequalities


from gurobipy import Model, GRB, quicksum, LinExpr
import networkx as nx
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
SD = 200  # Shut-down cost
a, b, c = 0.00211, 16.5, 680  # Quadratic cost coefficients

# Electricity prices for each time period
np.random.seed(42)
price = np.random.uniform(0, 44, T)


def create_model(relax=False):
    model = Model("SelfSchedulingUC")

    # Decision variables
    x = model.addVars(T, name="x")
    y = model.addVars(T, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="y")
    u = model.addVars(T, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="u")

    # Objective function
    model.setObjective(quicksum(price[t] * x[t] - (a * x[t] * x[t] + b * x[t] + c * y[t]) for t in range(T)) -
                       quicksum(SU * u[t] + SD * (y[t - 1] - y[t] + u[t]) for t in range(1, T)),
                       GRB.MAXIMIZE)

    # Constraints
    for t in range(T):
        model.addConstr(x[t] >= 0, f"GenLowerBound_{t}")
        model.addConstr(x[t] <= C_max * y[t], f"GenUpperBound_{t}")
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


cuts = 0


# Callback function implementing the separation approach for inequality (29)
def mycallback(model, where):
    global cuts
    if where == GRB.Callback.MIP:
        # Get the current solution values
        x_vals = model.cbGetSolution(model._x)
        y_vals = model.cbGetSolution(model._y)
        u_vals = model.cbGetSolution(model._u)

        for t in range(L + 1, T):
            G = nx.DiGraph()

            # Node set V: origin 'o', destination 'd', and time indices
            G.add_node("o")
            G.add_node("d")
            for m in range(t - L, t + 1):
                G.add_node(m)

            G.add_edge("o", t, weight=V_start * y_vals[t] - x_vals[t])

            m = min(t - L - 1, int((C_max - V_start) / V_up) - L + 1)
            if m >= 0 and t - m >= 0:
                weight_tmd = (C_max - V_start - (m + L - 1) * V_up) * (
                            y_vals[t] - sum(u_vals[t - m - j] for j in range(L)))
                G.add_edge(t - m, "d", weight=weight_tmd)

            for i in range(t - L + 1, t):
                if i > 8 and i < 1992:
                    weight_rs = V_up * sum(
                        y_vals[i + k] - sum(u_vals[i + k - j] for j in range(L)) for k in range(1, L))
                    G.add_edge(i, t, weight=weight_rs)

            # Find the shortest path from 'o' to 'd'
            try:
                shortest_path = nx.shortest_path(G, source="o", target="d", weight='weight')
                shortest_path_cost = nx.shortest_path_length(G, source="o", target="d", weight='weight')

                # If the shortest path cost is negative, add user cut
                if shortest_path_cost < 0:
                    cut_expr = LinExpr()
                    cut_expr += V_start * y_vals[t] - x_vals[t]
                    model.cbCut(cut_expr <= 0)
                    cuts += 1
            except nx.NetworkXNoPath:
                pass


# Solve MILP model
milp_model, x_milp, y_milp, u_milp = create_model(relax=False)
milp_model._x = x_milp
milp_model._y = y_milp
milp_model._u = u_milp
milp_model.setParam(GRB.Param.PreCrush, 1)

start = time.time()
milp_model.optimize()
end = time.time()
total_time_milp = end - start
Z_MILP = milp_model.ObjVal

# Solve LP relaxation model
lp_model, x_lp, y_lp, u_lp = create_model(relax=True)
start = time.time()
lp_model.optimize(mycallback)
end = time.time()
total_time_lp = end - start
Z_LP = lp_model.ObjVal

# Calculate integrality gap
integrality_gap = (Z_LP - Z_MILP) / Z_LP * 100

# Display results
print("\nMILP Solution:")
print(f"Optimal MILP Objective Value: {Z_MILP:.2f}")
print(f"Total Time for MILP: {total_time_milp:.2f} seconds")

print("\nLP Relaxation Solution:")
print(f"Optimal LP Relaxation Objective Value: {Z_LP:.2f}")
print(f"Total Time for LP Relaxation: {total_time_lp:.2f} seconds")

print(f"\nIntegrality Gap: {integrality_gap:.2f}%")
print(f"Total Number of User Cuts Added: {cuts}")
for t in range(T):
    print(f"Time {t}: x = {x_milp[t].X:.2f}, y = {y_milp[t].X}, u = {u_milp[t].X}")
