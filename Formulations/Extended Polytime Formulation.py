import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Parameters
T = 2000  # Time horizon
L = 8     # Minimum-up time
ell = 8   # Minimum-down time
C_max = 455  # Maximum generation (MW)
C_min = 150  # Minimum generation (MW)
V_up = 91    # Ramp-up limit (MW)
V_start = 180  # Ramp-down limit (MW)
SU = 2000    # Start-up cost
SD = 1000    # Shut-down cost
b, c = 16.19, 1000  # Quadratic cost coefficients

np.random.seed(42)
price = np.random.uniform(0, 35, T)


Q = calculate_Q(C_min, V_up, V_start, C_max, T)

# Sets
S = [0, 1, 2]  # Example set of states
S_successors = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1]
}  # Successors for each state

# Create a Gurobi model
def create_model():
    model = gp.Model("Deterministic_Single_UC")

    # Variables
    x = model.addVars(T, vtype=GRB.CONTINUOUS, name="x")  # Generation output
    y = model.addVars(T, vtype=GRB.BINARY, name="y")      # On/Off status
    u = model.addVars(T, vtype=GRB.BINARY, name="u")      # Start-up status
    w = model.addVars(T, S, S, vtype=GRB.BINARY, name="w")  # State transition variables

    # Objective Function
    def generation_cost(t):
        if t == 0:
            return SU * u[t] + SD * u[t] + (price[t] * x[t]) - (b * x[t] + c * y[t])
        else:
            return SU * u[t] + SD * (y[t-1] - y[t] + u[t]) + (price[t] * x[t]) - (b * x[t] + c * y[t])

    model.setObjective(
        gp.quicksum(generation_cost(t) for t in range(T)),
        GRB.MINIMIZE
    )

    # Constraint (16): Generation output must be in Q
    for t in range(T):
        model.addConstr(x[t] >= min(Q), name=f"gen_lower_bound_{t}")
        model.addConstr(x[t] <= max(Q), name=f"gen_upper_bound_{t}")

    # Constraint (20b): Initial state constraint
    for j in S_successors[1]:
        model.addConstr(w[0, 1, j] == 1, name=f"initial_state_{j}")

    # Constraint (20c): State transition constraint for non-terminal states
    for i in S:
        if i != 2:
            model.addConstr(gp.quicksum(w[0, i, j] for j in S_successors[i]) == 0, name=f"state_transition_{i}")

    # Constraint (20d): Flow conservation constraint
    for t in range(1, T):
        for i in S:
            model.addConstr(
                gp.quicksum(w[t, i, j] for j in S_successors[i]) - gp.quicksum(w[t-1, k, i] for k in S) == 0,
                name=f"flow_conservation_{t}_{i}"
            )

    # Additional Inequalities from Theorem 3
    for t in range(T):
        for i in S:
            for j in S_successors[i]:
                model.addConstr(x[t] <= 30 * w[t, i, j], name=f"ineq_x_upper_{t}_{i}_{j}")
                model.addConstr(y[t] <= w[t, i, j], name=f"ineq_y_upper_{t}_{i}_{j}")
                model.addConstr(u[t] <= w[t, i, j], name=f"ineq_u_upper_{t}_{i}_{j}")

    return model, x, y, u, w

# Create and optimize the model
model, x, y, u, w = create_model()
model.optimize()

# Display the results
print("Calculated set Q:", Q)

if model.status == GRB.OPTIMAL:
    print("\nOptimal Objective Value:", model.objVal)
    for t in range(T):
        print(f"Time {t}: x = {x[t].X}, y = {y[t].X}, u = {u[t].X}")
else:
    print("No optimal solution found.")
