import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 데이터 로드
df = pd.read_excel('C:\Contest\Cj\Result.xlsx')

# 목적지별 좌표 사전 생성
destination_coords = df.dropna(subset=["Destination", "Longitude", "Latitude"])\
    .drop_duplicates(subset=["Destination"])\
    .set_index("Destination")[["Longitude", "Latitude"]].to_dict(orient="index")

# 차량별 경로 시각화
plt.figure(figsize=(10, 8))

vehicle_ids = df["Vehicle_ID"].dropna().unique()
for vehicle_id in vehicle_ids:
    vehicle_df = df[df["Vehicle_ID"] == vehicle_id].sort_values("Route_Order")
    xs, ys = [], []
    for dest in vehicle_df["Destination"]:
        if dest in destination_coords:
            xs.append(destination_coords[dest]["Longitude"])
            ys.append(destination_coords[dest]["Latitude"])
    plt.plot(xs, ys, marker="o", label=f"Vehicle {int(vehicle_id)}")

plt.title("VRP 경로 시각화")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid()
plt.show()