import json
from collections import defaultdict, Counter
from ortools.constraint_solver import pywrapcp, routing_enums_pb2 
from ortools.sat.python import cp_model
import pandas as pd
from datetime import datetime


start_time = datetime.now()
print(f"start time: {start_time}")

def ip(b , capacity_matrix):
    model = cp_model.CpModel()
 
    x = [model.NewIntVar(0, 10, f'x{i}') for i in range(8)]

    #
    for i in range(3):
        model.Add(sum(capacity_matrix[j][i] * x[j] for j in range(len(x))) >= b[i])


    model.Minimize(sum(x))

    #
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    #
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(8):
            print(f"x{i} = {solver.Value(x[i])}")
    else:
        print(" ")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = [solver.Value(var) for var in x]
        return solution
    else:
        return "FUCK "
# ===================== Data Load =====================
def load_json(path="data.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_original_distance(path="distance-data.txt"):
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    df["ORIGIN"] = df["ORIGIN"].astype(str)
    df["DESTINATION"] = df["DESTINATION"].astype(str)
    nodes = sorted(set(df["ORIGIN"]) | set(df["DESTINATION"]))
    base_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    mat = [[0] * n for _ in range(n)]
    for _, row in df.iterrows():
        i = base_to_idx[row["ORIGIN"]]
        j = base_to_idx[row["DESTINATION"]]
        mat[i][j] = int(row["DISTANCE_METER"])
    return mat, base_to_idx, {i: node for node, i in base_to_idx.items()}

# ===================== Order Classification =====================
def classify_orders(data):
    type_map = {(30,40,30):0, (30,50,40):1, (50,60,50):2}
    orders = []
    for o in data["orders"]:
        d = o["dimension"]
        key = (d["width"], d["length"], d["height"])
        if key in type_map:
            orders.append((type_map[key], o))
    return orders
# =======================================================================save excel
def save_to_excel(packed_list, vehicle_routes, box_label, data):
    destination_coords = {
        d["destination_id"]: (d["location"]["latitude"], d["location"]["longitude"])
        for d in data["destinations"]
    }
    depot_lat = data["depot"]["location"]["latitude"]
    depot_lon = data["depot"]["location"]["longitude"]

    records = []
    vehicle_id = 0
    for route, packed in zip(vehicle_routes, packed_list):
        records.append({
            "Vehicle_ID": vehicle_id,
            "Route_Order": 1,
            "Destination": "Depot",
            "Order_Number": None,
            "Box_ID": None,
            "Stacking_Order": None,
            "Lower_Left_X": None,
            "Lower_Left_Y": None,
            "Lower_Left_Z": None,
            "Longitude": None,
            "Latitude": None,
            "Box_Width": None,
            "Box_Length": None,
            "Box_Height": None
        })
        n=len(packed)
        for i, (order, (x, y, z, dx, dy, dz)) in enumerate(packed, start=1):
            lat, lon = destination_coords[order["destination"]]
            records.append({
                "Vehicle_ID": vehicle_id,
                "Route_Order": i+1,
                "Destination": order["destination"],
                "Order_Number": order["order_number"],
                "Box_ID": order["box_id"],
                "Stacking_Order": i,
                "Lower_Left_X": x,
                "Lower_Left_Y": y,
                "Lower_Left_Z": z,
                "Longitude": lon,
                "Latitude": lat,
                "Box_Width": dx,
                "Box_Length": dy,
                "Box_Height": dz
            })

        records.append({
            "Vehicle_ID": vehicle_id,
            "Route_Order": len(packed) + 2,
            "Destination": "Depot",
            "Order_Number": None,
            "Box_ID": None,
            "Stacking_Order": None,
            "Lower_Left_X": None,
            "Lower_Left_Y": None,
            "Lower_Left_Z": None,
            "Longitude": None,
            "Latitude": None,
            "Box_Width": None,
            "Box_Length": None,
            "Box_Height": None
        })
        vehicle_id += 1

    df = pd.DataFrame(records)
    df.to_excel("Result.xlsx", index=False)
# ===================== Build Split Nodes =====================
def build_split_problem(orig_mat, base_to_idx, orders):
    # aggregate demand per (base, type)
    demand = defaultdict(int)
    for t, o in orders:
        demand[(o["destination"], t)] += 1

    # build split node names
    split_names = ["Depot"] + [f"{dest}_{t}" for (dest, t), cnt in demand.items()]
    node_to_idx = {name: i for i, name in enumerate(split_names)}
    idx_to_name = {i: name for name, i in node_to_idx.items()}

    # initialize split distance matrix
    size = len(split_names)
    mat = [[0] * size for _ in range(size)]
    for i in range(size):
        ni = idx_to_name[i]
        # extract base node name by splitting on last underscore
        if ni == "Depot":
            base_i = "Depot"
        else:
            parts = ni.rsplit('_', 1)
            if len(parts) == 2:
                base_i, _ = parts
            else:
                base_i = ni
        if base_i not in base_to_idx:
            raise KeyError(f"Base node '{base_i}' not found in distance matrix")
        idx_base_i = base_to_idx[base_i]
        for j in range(size):
            nj = idx_to_name[j]
            if nj == "Depot":
                base_j = "Depot"
            else:
                parts = nj.rsplit('_', 1)
                if len(parts) == 2:
                    base_j, _ = parts
                else:
                    base_j = nj
            if base_j not in base_to_idx:
                raise KeyError(f"Base node '{base_j}' not found in distance matrix")
            idx_base_j = base_to_idx[base_j]
            # same base -> zero distance for split-nodes
            if base_i == base_j:
                mat[i][j] = 0
            else:
                mat[i][j] = orig_mat[idx_base_i][idx_base_j]

    # demand dicts per type
    dem0, dem1, dem2 = {}, {}, {}
    for (dest, t), cnt in demand.items():
        name = f"{dest}_{t}"
        if t == 0: dem0[name] = cnt
        elif t == 1: dem1[name] = cnt
        else: dem2[name] = cnt

    return mat, node_to_idx, idx_to_name, dem0, dem1, dem2
# =======================================pack
def pack_boxes(orders, route):

    dx = (30 , 50 , 50)
    dz = (30 , 30 , 60)
    dy = (40 , 40 , 50)            # dx[0]은    0번박스(s)의 x크기 
    max_y=[6 ,6 ,3]    # 박스 타입별 최대갯수
    box_type = None
    idx_max = (42,42,15)
    pushed_X = 0
    present_x_coord_by_box_type = [0,0,0]
    count_idx=[0,0,0]
    pos = []

    for dest in reversed(route):
        if dest == "Depot":
            continue
        for  o_type , o in orders:
            if o["destination"] == dest[:-2]  and o_type == int(dest[-1]): 
                box_type = int(dest[-1])
                if count_idx[box_type] == 0:
                    present_x_coord_by_box_type[box_type] = pushed_X
                    pushed_X += dx[box_type]
                cx , cy , cz = present_x_coord_by_box_type[box_type], (count_idx[box_type] // max_y[box_type]) * dy[box_type], (count_idx[box_type] % max_y[box_type])*dz[box_type]
                pos.append((o, (cx, cy, cz, dx[box_type], dy[box_type], dz[box_type])))
                count_idx[box_type] +=1
                if count_idx[box_type] >= idx_max[box_type]:
                    count_idx[box_type] = 0
    return pos

# ===================== VRP Solver =====================
def solve_vrp(mat, node_to_idx, idx_to_name, dem0, dem1, dem2,
              vehicle_types, capacity_matrix, time_limit=200):
    num_veh = len(vehicle_types)
    depot   = node_to_idx["Depot"]
    manager = pywrapcp.RoutingIndexManager(len(mat), num_veh, depot)
    routing_params = pywrapcp.DefaultRoutingModelParameters()
    routing = pywrapcp.RoutingModel(manager, routing_params)

    transit_cb = routing.RegisterTransitCallback(
        lambda i, j: mat[manager.IndexToNode(i)][manager.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    def mk_cb(demand_dict):
        return lambda idx: demand_dict.get(idx_to_name[manager.IndexToNode(idx)], 0)

    cap0 = [capacity_matrix[t][0] for t in vehicle_types]
    cap1 = [capacity_matrix[t][1] for t in vehicle_types]
    cap2 = [capacity_matrix[t][2] for t in vehicle_types]

    cb0 = routing.RegisterUnaryTransitCallback(mk_cb(dem0))
    cb1 = routing.RegisterUnaryTransitCallback(mk_cb(dem1))
    cb2 = routing.RegisterUnaryTransitCallback(mk_cb(dem2))

    routing.AddDimensionWithVehicleCapacity(cb0, 0, cap0, True, "Box0")
    routing.AddDimensionWithVehicleCapacity(cb1, 0, cap1, True, "Box1")
    routing.AddDimensionWithVehicleCapacity(cb2, 0, cap2, True, "Box2")

    # (4) 검색 파라미터
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    # search_params.log_search = True
    search_params.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return "Fuck"
    routes = []
    for v in range(num_veh):
        idx = routing.Start(v)
        path = []
        while not routing.IsEnd(idx):
            path.append(idx_to_name[ manager.IndexToNode(idx) ])
            idx = solution.Value(routing.NextVar(idx))
        path.append("Depot")
        routes.append(path)
    return routes


if __name__ == "__main__":

    data = load_json()
    orders = classify_orders(data)
    orig_mat, base_to_idx, idx_to_base = load_original_distance()
    sim_orders = [
        (box_type, o["destination"])
        for box_type, o in orders
    ]

    mat, nid, idn, dem0, dem1, dem2 = build_split_problem(orig_mat, base_to_idx, orders)
    capacity_matrix = [[84,84,0],[0,84,15],[0,42,30],[84,0,30],[210,0,0],[0,126,0],[0,0,45], [84, 42 , 15] ]
    b=list([sum(dem0.values()) , sum(dem1.values()), sum(dem2.values())])
    ip_sol = ip(b , capacity_matrix)
    vehicle_types =[]
    for idx, count in enumerate(ip_sol):
        if count > 0:
            vehicle_types.extend([idx] * count)
    routes = solve_vrp(mat, nid, idn, dem0, dem1, dem2, vehicle_types, capacity_matrix)
    all_packed = []
    for route in routes:
        packed = pack_boxes(orders , route)
        all_packed.append(packed)

    save_to_excel(all_packed, routes, "All", data)


