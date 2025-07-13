import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import MaxRuntime
from alns.select import RandomSelect
import numpy as np
import random
import math
import time

# 랜덤 시드 고정 (재현성)
random.seed(42)
np.random.seed(42)

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
def solve_cvrp_alns(distance_matrix, node_to_index, index_to_node, orders, vehicle_capacity, max_seconds=380, init_temp=800, cooling=0.96, destroy_strength=0.13, decay=0.96):
    import copy
    depot = "Depot"
    depot_idx = node_to_index[depot]
    demands = defaultdict(int)
    for o in orders:
        demands[o["destination"]] += 1
    customer_nodes = [node for node in demands.keys()]
    
    def initial_solution():
        # Greedy insertion 초기화
        unassigned = set(customer_nodes)
        routes = []
        while unassigned:
            route = []
            load = 0
            prev = depot
            while unassigned:
                # 가장 가까운 노드 선택
                next_node = min(unassigned, key=lambda n: distance_matrix[node_to_index[prev]][node_to_index[n]])
                if load + demands[next_node] > vehicle_capacity:
                    break
                route.append(next_node)
                load += demands[next_node]
                prev = next_node
                unassigned.remove(next_node)
            if route:
                routes.append(route)
        return routes

    def calc_cost(routes):
        total_dist = 0
        for route in routes:
            if not route:
                continue
            prev = depot
            for dest in route:
                total_dist += distance_matrix[node_to_index[prev]][node_to_index[dest]]
                prev = dest
            total_dist += distance_matrix[node_to_index[prev]][depot_idx]
        return len(routes)*150000 + total_dist*500

    # ----------- 파괴 연산자 -----------
    def destroy_shaw(routes, n_remove):
        # 유사 노드(거리 기준) 그룹을 파괴_strength만큼 제거
        routes = copy.deepcopy(routes)
        all_nodes = [dest for route in routes for dest in route]
        if len(all_nodes) <= 1:
            return routes, []
        removed = set()
        # seed node 고정
        seed = random.choice(all_nodes)
        removed.add(seed)
        while len(removed) < n_remove:
            candidates = [n for n in all_nodes if n not in removed]
            if not candidates:
                break
            # 유사도: 거리 + 수요
            last = list(removed)[-1]
            sim = [(n, distance_matrix[node_to_index[last]][node_to_index[n]] + abs(demands[last] - demands[n])) for n in candidates]
            sim.sort(key=lambda x: x[1])
            removed.add(sim[0][0])
        new_routes = []
        for route in routes:
            new_route = [dest for dest in route if dest not in removed]
            if new_route:
                new_routes.append(new_route)
        return new_routes, list(removed)

    def destroy_worst(routes, n_remove):
        # 비용 증가가 큰 노드 n개 제거
        routes = copy.deepcopy(routes)
        all_nodes = [dest for route in routes for dest in route]
        if len(all_nodes) <= 1:
            return routes, []
        # 각 노드의 제거 시 cost 감소량 계산
        deltas = []
        for node in all_nodes:
            cost_before = calc_cost(routes)
            temp_routes = [list(r) for r in routes]
            for r in temp_routes:
                if node in r:
                    r.remove(node)
            cost_after = calc_cost(temp_routes)
            deltas.append((node, cost_before - cost_after))
        deltas.sort(key=lambda x: -x[1])
        removed = set([n for n, _ in deltas[:n_remove]])
        new_routes = []
        for route in routes:
            new_route = [dest for dest in route if dest not in removed]
            if new_route:
                new_routes.append(new_route)
        return new_routes, list(removed)

    def destroy_route(routes, n_remove):
        # 전체 route 중 하나를 통째로 제거
        routes = copy.deepcopy(routes)
        if len(routes) <= 1:
            return routes, []
        n = min(n_remove, len(routes))
        idxs = random.sample(range(len(routes)), n)
        removed = []
        new_routes = []
        for i, r in enumerate(routes):
            if i in idxs:
                removed.extend(r)
            else:
                new_routes.append(r)
        return new_routes, removed

    destroy_ops = [destroy_shaw, destroy_worst, destroy_route]
    destroy_names = ["shaw", "worst", "route"]

    # ----------- 복구 연산자 -----------
    def repair_greedy(routes, removed):
        # 가장 비용 증가가 적은 곳에 하나씩 삽입
        routes = copy.deepcopy(routes)
        loads = [sum(demands[n] for n in r) for r in routes]
        for dest in removed:
            d = demands[dest]
            best_i, best_pos, best_incr = None, None, float('inf')
            for i, route in enumerate(routes):
                if loads[i] + d > vehicle_capacity:
                    continue
                for pos in range(len(route)+1):
                    test_route = route[:pos] + [dest] + route[pos:]
                    dist = 0
                    prev = depot
                    for nd in test_route:
                        dist += distance_matrix[node_to_index[prev]][node_to_index[nd]]
                        prev = nd
                    dist += distance_matrix[node_to_index[prev]][depot_idx]
                    incr = dist - (
                        distance_matrix[node_to_index[depot]][node_to_index[route[0]]] +
                        sum(distance_matrix[node_to_index[route[i-1]]][node_to_index[route[i]]] for i in range(1, len(route))) +
                        distance_matrix[node_to_index[route[-1]]][depot_idx] if route else 0
                    )
                    if incr < best_incr:
                        best_i, best_pos, best_incr = i, pos, incr
            if best_i is not None:
                routes[best_i].insert(best_pos, dest)
                loads[best_i] += d
            else:
                routes.append([dest])
                loads.append(d)
        return routes

    def repair_regret2(routes, removed):
        # regret-2: 두 번째로 좋은 위치와의 비용 차이가 큰 것부터 삽입
        routes = copy.deepcopy(routes)
        loads = [sum(demands[n] for n in r) for r in routes]
        unassigned = set(removed)
        while unassigned:
            regrets = []
            for dest in unassigned:
                d = demands[dest]
                incrs = []
                for i, route in enumerate(routes):
                    if loads[i] + d > vehicle_capacity:
                        continue
                    for pos in range(len(route)+1):
                        test_route = route[:pos] + [dest] + route[pos:]
                        dist = 0
                        prev = depot
                        for nd in test_route:
                            dist += distance_matrix[node_to_index[prev]][node_to_index[nd]]
                            prev = nd
                        dist += distance_matrix[node_to_index[prev]][depot_idx]
                        incr = dist - (
                            distance_matrix[node_to_index[depot]][node_to_index[route[0]]] +
                            sum(distance_matrix[node_to_index[route[i-1]]][node_to_index[route[i]]] for i in range(1, len(route))) +
                            distance_matrix[node_to_index[route[-1]]][depot_idx] if route else 0
                        )
                        incrs.append((incr, i, pos))
                if incrs:
                    incrs.sort()
                    regret = incrs[1][0] - incrs[0][0] if len(incrs) > 1 else 0
                    regrets.append((regret, incrs[0][0], dest, incrs[0][1], incrs[0][2]))
                else:
                    regrets.append((-1, float('inf'), dest, None, None))
            regrets.sort(reverse=True)
            _, _, dest, best_i, best_pos = regrets[0]
            d = demands[dest]
            if best_i is not None:
                routes[best_i].insert(best_pos, dest)
                loads[best_i] += d
            else:
                routes.append([dest])
                loads.append(d)
            unassigned.remove(dest)
        return routes

    repair_ops = [repair_greedy, repair_regret2]
    repair_names = ["greedy", "regret2"]

    # ----------- 2-opt (best 갱신 시만) -----------
    def two_opt(route):
        # route: depot 제외
        best = route[:]
        best_dist = route_distance(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-1):
                for j in range(i+1, len(best)):
                    if j-i < 2:
                        continue
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best = new_route
                        best_dist = new_dist
                        improved = True
        return best

    def route_distance(route):
        if not route:
            return 0
        prev = depot
        dist = 0
        for dest in route:
            dist += distance_matrix[node_to_index[prev]][node_to_index[dest]]
            prev = dest
        dist += distance_matrix[node_to_index[prev]][depot_idx]
        return dist

    # ----------- ALNS 메인 루프 -----------
    current = initial_solution()
    best = [list(r) for r in current]
    best_cost = calc_cost(best)
    temp = init_temp
    start_time = time.time()
    iter_count = 0
    destroy_ops = [destroy_shaw, destroy_worst, destroy_route]
    repair_ops = [repair_greedy, repair_regret2]
    destroy_weights = [1.0 for _ in destroy_ops]
    repair_weights = [1.0 for _ in repair_ops]
    destroy_scores = [0.0 for _ in destroy_ops]
    repair_scores = [0.0 for _ in repair_ops]
    decay = 0.96
    destroy_strength = 0.13
    while time.time() - start_time < max_seconds:
        # 연산자 softmax 확률
        d_probs = np.array(destroy_weights) / sum(destroy_weights)
        r_probs = np.array(repair_weights) / sum(repair_weights)
        d_idx = np.random.choice(len(destroy_ops), p=d_probs)
        r_idx = np.random.choice(len(repair_ops), p=r_probs)
        destroy = destroy_ops[d_idx]
        repair = repair_ops[r_idx]
        n_remove = max(1, int(len(customer_nodes) * destroy_strength))
        partial, removed = destroy(current, n_remove)
        candidate = repair(partial, removed)
        # 용량 초과 경로 제거
        candidate = [r for r in candidate if sum(demands[n] for n in r) <= vehicle_capacity]
        cand_cost = calc_cost(candidate)
        delta = cand_cost - calc_cost(current)
        accepted = False
        is_best = False
        is_better = False
        if cand_cost < best_cost:
            # 각 route에 2-opt 적용
            candidate2 = []
            for r in candidate:
                if len(r) > 3:
                    r2 = two_opt(r)
                    candidate2.append(r2)
                else:
                    candidate2.append(r)
            candidate = candidate2
            cand_cost = calc_cost(candidate)
            delta = cand_cost - calc_cost(current)
            if cand_cost < best_cost:
                best = [list(r) for r in candidate]
                best_cost = cand_cost
                is_best = True
        if delta < 0:
            current = [list(r) for r in candidate]
            accepted = True
            if not is_best:
                is_better = True
        elif random.random() < math.exp(-delta/temp):
            current = [list(r) for r in candidate]
            accepted = True
        # 점수 부여
        if is_best:
            destroy_scores[d_idx] += 33
            repair_scores[r_idx] += 33
        elif is_better:
            destroy_scores[d_idx] += 9
            repair_scores[r_idx] += 9
        elif accepted:
            destroy_scores[d_idx] += 3
            repair_scores[r_idx] += 3
        # else: 0점
        if iter_count % 50 == 0 and iter_count > 0:
            destroy_weights = [decay * w + s for w, s in zip(destroy_weights, destroy_scores)]
            repair_weights = [decay * w + s for w, s in zip(repair_weights, repair_scores)]
            destroy_scores = [0.0 for _ in destroy_ops]
            repair_scores = [0.0 for _ in repair_ops]
        temp *= cooling
        iter_count += 1
    print(f"ALNS 반복 횟수: {iter_count}")
    # depot 포함 경로 반환
    routes = []
    for route in best:
        if not route:
            continue
        full_route = [depot] + route + [depot]
        routes.append(full_route)
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
        route_plan = solve_cvrp_alns(matrix, n2i, i2n, orders, capacities[t])
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