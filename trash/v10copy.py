import json
import pandas as pd
from collections import defaultdict,Counter
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ===================== Load Data =====================
def load_json(path="C:\Contest\Cj\data.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_distance_matrix(path="C:\Contest\Cj\distance-data.txt"):
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
    orders_by_type = []
    for order in data["orders"]:
        dim = order["dimension"]
        key = (dim["width"], dim["length"], dim["height"])
        if key in box_type_map:
            _, idx = box_type_map[key]
            order["box_type_index"] = idx
            orders_by_type.append( (idx, order) )

    return orders_by_type
 
# ===================== VRP Solver =====================
def solve_vrp(distance_matrix, node_to_index, index_to_node, orders_by_type, vehicle_capacity):
    depot = "Depot"
    depot_idx = node_to_index[depot]
    box_capacities = {
    0: 1,
    1: 2,
    2: 4
}

    demands = defaultdict(int)
    for o_type , o in orders_by_type:
        demands[o["destination"]] += box_capacities[o_type]

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
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 300

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
#=============================rest box packing====================
def rest_box_packing(rest_boxes, last_positions,full_check,box_dims):
    pos_list={0:[],1:[],2:[]}
    pos_ground = {}
    pos_ground[0] = [last_positions[0][0],last_positions[0][1] + 50,0]
    pos_ground[1] = [last_positions[1][0],last_positions[1][1] + 40,0]
    pos_ground[2] = [100, last_positions[2][1] + 40,0]

    print("4: last_positions")
    last_positions[2][0] = 100
    print(last_positions)

    for box_type , box_count in rest_boxes.items():
        
        pos_ground[0] = [last_positions[0][0],last_positions[0][1] + 50,0]
        if box_type==2:
            pos_ground[1] = [last_positions[1][0],last_positions[1][1] + 50,0] 
            pos_ground[2] = [100,last_positions[2][1] + 50,0] 
        else:
            pos_ground[1] = [last_positions[1][0],last_positions[1][1] + 40,0] 
            pos_ground[2] = [100,last_positions[2][1] + 40,0] 
        # print("pos_ground")
        # print(pos_ground)

        while box_count > 0:
            if box_type==2:
                if full_check[1]==0:
                    if last_positions[1][1] + box_dims[box_type][1] <=280:
                        if last_positions[1][2]+ box_dims[box_type][2] <=180:
                            pos_list[2].append(last_positions[1][:])
                            last_positions[1][2]+=box_dims[box_type][2]
                            box_count-=1
                            print("2번 박스 1번위치")
                            print(f" 나머지 적재 디버깅용 last positions: {last_positions[1]}")
                            print(f" 나머지 적재 디버깅용 pos_list: {pos_list}")
                        else:
                            last_positions[1] = pos_ground[1][:]
                            pos_ground[1][1] += box_dims[box_type][1]
                    else:
                        full_check[1]=1
                        print("1번칸 꽉 찼습니다")

                elif full_check[2]==0:
                    if last_positions[2][1] + box_dims[box_type][1] <=280:
                        if last_positions[2][2] + box_dims[box_type][2] <=180:
                            pos_list[2].append(last_positions[2][:])
                            last_positions[2][2]+= box_dims[box_type][2]
                            box_count-=1
                            print("2번 박스 2번위치")
                            print(f" 나머지 적재 디버깅용 last positions: {last_positions[2]}")
                            print(f" 나머지 적재 디버깅용 pos_list: {pos_list}")
                        else:
                            last_positions[2] = pos_ground[2][:]
                            pos_ground[2][1] += box_dims[box_type][1]
                    else:
                        full_check[2]=1
                        print("2번칸 꽉 찼습니다")


            elif box_type==1:
                if full_check[0]==0:
                    if last_positions[0][1] + box_dims[box_type][1] <=280:
                        if last_positions[0][2]+ box_dims[box_type][2] <=180:
                            pos_list[1].append(last_positions[0][:])
                            last_positions[0][2]+=box_dims[box_type][2]
                            box_count-=1
                        else:
                            last_positions[0] = pos_ground[0][:]
                            pos_ground[0][1] += box_dims[box_type][1]
                    else:
                        full_check[0]=1
                
                elif full_check[2]==0:
                    if last_positions[2][1] + box_dims[box_type][1] <=280:
                        if last_positions[2][2]+ box_dims[box_type][2] <=180:
                            pos_list[1].append(last_positions[2][:])
                            last_positions[2][2]+=box_dims[box_type][2]
                            box_count-=1
                        else:
                            last_positions[2] = pos_ground[2][:]
                            pos_ground[2][1] += box_dims[box_type][1]
                    else:
                        full_check[2]=1

            elif box_type==0:
                if full_check[0]==0:
                    print("오류발생")
                elif full_check[1]==0:    
                    print("오류발생")
    return(pos_list)

# ===================== Packing Logic =====================
def pack_boxes(orders_by_type,destination_to_order_info, route, box_dims, container_dim):
    ex_pos={0:[],1:[],2:[]}

    box_type_counts = Counter()



    for dest in route:
        if dest == "Depot":
            continue
        for box_type, _ in destination_to_order_info.get(dest, []):
            box_type_counts[box_type] += 1
    box_type_counts = dict(sorted(box_type_counts.items(), key=lambda x: x[0], reverse=True))
    print("1: box_type_counts")
    print(box_type_counts)


    if box_type_counts[0]>84:
        print("비상 특이케이스라서 따로 로직 작성해야됨")
        # 나중에 추가


    full_check=[0,0,0]
    rest_boxes={}
    last_positions={}
    for box_type , box_count in box_type_counts.items():
        if box_type==2:
            if box_count> 15:
                full_check[0]=1
                rest_boxes[2]=box_count-15
                for a in range(15):
                    ex_pos[2].append((0,(a//3)*50,(a%3)*60))
                last_positions[0] = [0,((a+1)//3)*50,((a+1)%3)*60]
            else:
                for a in range(box_count):
                    ex_pos[2].append((0,(a//3)*50,(a%3)*60))
                last_positions[0] = [0,((a+1)//3)*50,((a+1)%3)*60]
                    
        elif box_type==1:
            if box_count>48:
                full_check[1]=1
                rest_boxes[1]=box_count-48
                for a in range(48):
                    ex_pos[1].append((50,(a//6)*40,(a%6)*30))
                last_positions[1] = [50,((a+1)//6)*40,((a+1)%6)*30]
            else:
                for a in range(box_count):
                    ex_pos[1].append((50,(a//6)*40,(a%6)*30))
                last_positions[1] = [50,((a+1)//6)*40,((a+1)%6)*30]
        elif box_type==0:
            if box_count>84 :
                full_check[2]=1
                rest_boxes[0]=box_count-84
                for a in range(84):
                    ex_pos[0].append((100+(a%2)*30,(a//12)*40,((a//2)%6)*30))
                last_positions[2] = [100+((a+1)%2)*30,((a+1)//12)*40,(((a+1)//2)%6)*30]
            else:
                for a in range(box_count):
                    ex_pos[0].append((100+(a%2)*30,(a//12)*40,((a//2)%6)*30))
                last_positions[2] = [100+((a+1)%2)*30,((a+1)//12)*40,(((a+1)//2)%6)*30]
    print(f"현재 적재상황 : {ex_pos}")
    print(f"2: 남은 박스들 : {rest_boxes}")
    print(f"3: 빈공간들 : {full_check}")
    rest_pos = rest_box_packing(rest_boxes, last_positions,full_check,box_dims)
    print("5:  나머지 적재")
    print(rest_pos)
    merged_pos = {}
    for k in set(ex_pos.keys()).union(rest_pos.keys()):
        merged_pos[k] = []
    
    # ex_pos에 해당 키가 있으면 추가
        if k in ex_pos:
            merged_pos[k].extend(ex_pos[k])
    
    # rest_pos에 해당 키가 있으면 추가
        if k in rest_pos:
            merged_pos[k].extend(tuple(i) for i in rest_pos[k])
    print(f"merged_pos = {merged_pos}")





    final_data = []
    pos_tracker = {0: 0, 1: 0, 2: 0}  # 박스타입별 좌표 인덱스 추적

    for dest in (route):
        if dest == "Depot":
            continue
        for box_type, box_info in destination_to_order_info.get(dest, []):
            coord = merged_pos[box_type][pos_tracker[box_type]]
            pos_tracker[box_type] += 1
            final_data.append((box_info, (*coord, *box_dims[box_type])))
    # print("final_data")
    # print("===================================================")
    # print(final_data)
    # print("===================================================")
    return final_data
# ===================== Save Excel =====================

# 수정된 save_to_excel 함수
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
        
        # 착지별 박스 분리
        dest_to_boxes = {}
        for order_data in packed:
            order_info, (x, y, z, dx, dy, dz) = order_data
            dest = order_info["destination"]
            if dest not in dest_to_boxes:
                dest_to_boxes[dest] = []
            dest_to_boxes[dest].append((order_info, (x, y, z, dx, dy, dz)))

        route_order = 1
        # 출발지 기록
        records.append({
            "Vehicle_ID": vehicle_id,
            "Route_Order": route_order,
            "Destination": "Depot",
            "Order_Number": None,
            "Box_ID": None,
            "Stacking_Order": None,
            "Lower_Left_X": None,
            "Lower_Left_Y": None,
            "Lower_Left_Z": None,
            "Longitude": depot_lon,
            "Latitude": depot_lat,
            "Box_Width": None,
            "Box_Length": None,
            "Box_Height": None
        })
        route_order += 1

        stacking_order = 0
        for dest in (route):
            if dest == "Depot":
                continue

            lat, lon = destination_coords[dest]

            for order_info, (x, y, z, dx, dy, dz) in dest_to_boxes.get(dest, []):
                stacking_order += 1
                records.append({
                    "Vehicle_ID": vehicle_id,
                    "Route_Order": route_order,
                    "Destination": dest,
                    "Order_Number": order_info["order_number"],
                    "Box_ID": order_info["box_id"],
                    "Stacking_Order": stacking_order,
                    "Lower_Left_X": x,
                    "Lower_Left_Y": y,
                    "Lower_Left_Z": z,
                    "Longitude": lon,
                    "Latitude": lat,
                    "Box_Width": dx,
                    "Box_Length": dy,
                    "Box_Height": dz
                })
            route_order += 1

        # 도착지 기록
        records.append({
            "Vehicle_ID": vehicle_id,
            "Route_Order": route_order,
            "Destination": "Depot",
            "Order_Number": None,
            "Box_ID": None,
            "Stacking_Order": None,
            "Lower_Left_X": None,
            "Lower_Left_Y": None,
            "Lower_Left_Z": None,
            "Longitude": depot_lon,
            "Latitude": depot_lat,
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
        0: (30, 40, 30),
        1: (50, 40, 30),
        2: (50, 50, 60)
    }
    # capacities = {
    #     0: 216,
    #     1: 126,
    #     2: 45
    # }

    all_packed = []
    all_routes = []
    # print("orders_by_type")
    # print(orders_by_type)


    route_plan = solve_vrp(matrix, n2i, i2n, orders_by_type, 180)
    print("route_plan++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(route_plan)
    # print(len(route_plan))
    # visited = set()
    # for vehicle_id, route in enumerate(route_plan):
    #     for dest in route:
    #         if dest == "Depot":
    #             continue
    #         if dest in visited:
    #             print(f"⚠️ 착지 {dest} 여러 차량이 방문!")
    #         visited.add(dest)

    packed_all = []
    destination_to_order_info = defaultdict(list)

    for box_type, box_info in orders_by_type:
        destination_to_order_info[box_info["destination"]].append( (box_type, box_info) )
    for route in route_plan:
        packed = pack_boxes(orders_by_type,destination_to_order_info, route, box_dims, truck_dim)
        packed_all.append(packed)
    print("==============packed_all")
    print(packed_all)
    save_to_excel(packed_all,route_plan , "All", data)


# print("=== 차량별 박스타입별 수량 및 총 적재용량 ===")
# print("==================== route plan")
# for n in route_plan:
#     print(n)
# for vehicle_id, route in enumerate(route_plan):
#     type_counts = {0: 0, 1: 0, 2: 0}
#     total_capacity = 0

#     for dest in route:
#         if dest == "Depot":
#             continue
#         for box_type, order in orders_by_type:
#             if order["destination"] == dest:
#                 type_counts[box_type] += 1
#                 total_capacity += box_capacities[box_type]

#     print(f"[차량 {vehicle_id}]")
#     for t in sorted(type_counts):
#         print(f"  박스타입 {t}: {type_counts[t]}개")
#     print(f"  총 적재용량: {total_capacity}\n")
