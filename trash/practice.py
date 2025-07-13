import json
import pandas as pd
from collections import defaultdict
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from datetime import datetime
# 시간 측정 시작
start_time = datetime.now()
print(f"start time: {start_time}")
# ===================== Load Data =====================
def load_json(path="data.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_distance_matrix(path="distance-data.txt"):
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    df["ORIGIN"] = df["ORIGIN"].astype(str)
    df["DESTINATION"] = df["DESTINATION"].astype(str)

    nodes = sorted(set(df["ORIGIN"]).union(set(df["DESTINATION"])))
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    n = len(nodes)
    matrix = [[0] * n for _ in range(n)]
    for _, row in df.iterrows():
        i = node_to_index[row["ORIGIN"]]
        j = node_to_index[row["DESTINATION"]]
        matrix[i][j] = int(row["DISTANCE_METER"])

    return matrix, node_to_index, index_to_node

# ===================== Classify Orders =====================
def classify_orders(data):
    box_type_map = {
        (30, 40, 30): ("S", 0),
        (30, 50, 40): ("M", 1),
        (50, 60, 50): ("L", 2)
    }
    orders_by_type = defaultdict(list)
    for order in data["orders"]:
        dim = order["dimension"]
        key = (dim["width"], dim["length"], dim["height"])
        if key in box_type_map:
            _, idx = box_type_map[key]
            order["box_type_index"] = idx
            orders_by_type[idx].append(order)
    return orders_by_type

# ===================== VRP Solver =====================
def solve_vrp(distance_matrix, node_to_index, index_to_node, orders, vehicle_capacity):
    depot = "Depot"
    depot_idx = node_to_index[depot]

    demands = defaultdict(int)
    for o in orders:
        demands[o["destination"]] += 1

    num_vehicles = 15  # 최대 차량 수 (유연하게 조정 가능)
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        dest = index_to_node[node]
        return demands.get(dest, 0)

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [vehicle_capacity]*num_vehicles, True, 'Capacity'
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    search_params.time_limit.seconds = 200

    solution = routing.SolveWithParameters(search_params)

    routes = []
    if solution:
        for v in range(num_vehicles):
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(index_to_node[node])
                index = solution.Value(routing.NextVar(index))
            route.append(index_to_node[manager.IndexToNode(index)])
            if len(route) > 2:
                routes.append(route)
    return routes

# ===================== Packing Logic =====================
def pack_boxes(orders, route, box_dim, container_dim):
    x_len, y_len, z_len = container_dim
    dx, dy, dz = box_dim

    pos = []
    col_count = 0
    max_row = x_len // dx
    max_layer = z_len // dz
    max_depth = y_len // dy

    for dest in reversed(route):
        if dest == "Depot":
            continue
        for o in orders:
            if o["destination"] == dest:
                cx = (col_count % max_row) * dx
                cz = (col_count // max_row % max_layer) * dz
                cy = (col_count // (max_row * max_layer)) * dy
                pos.append((o, (cx, y_len - dy - cy, cz, dx, dy, dz)))
                col_count += 1
    return pos

# ===================== Save Excel =====================
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

# ===================== Run All =====================
if __name__ == "__main__":
    data = load_json()
    orders_by_type = classify_orders(data)
    matrix, n2i, i2n = load_distance_matrix()

    truck_dim = (160, 280, 180)
    box_dims = {
        0: (40, 30, 30),
        1: (50, 40, 30),
        2: (50, 50, 60)
    }
    capacities = {
        0: 216,
        1: 126,
        2: 45
    }

    all_packed = []
    all_routes = []

    for t in [0, 1, 2]:
        orders = orders_by_type[t]
        route_plan = solve_vrp(matrix, n2i, i2n, orders, capacities[t])
        packed_all = []
        for route in route_plan:
            packed = pack_boxes(orders, route, box_dims[t], truck_dim)
            packed_all.append(packed)
        all_packed += packed_all
        all_routes += route_plan

    save_to_excel(all_packed, all_routes, "All", data)

    # 시간 측정 끝
    end_time = datetime.now()
    print(f"end time: {end_time}")
    print(f"run time: {end_time - start_time}")