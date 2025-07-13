import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ===================== 데이터 로딩 =====================
df = pd.read_excel("C:\Contest\Cj\Result.xlsx")  # 결과 엑셀 파일 경로

df_valid = df[df["Box_ID"].notna()]  # 실제 박스만 필터링

# ===================== 전체 시각화 함수 =====================
def visualize_all_vehicles(df, truck_w=160, truck_l=280, truck_h=180):
    vehicle_ids = sorted(df["Vehicle_ID"].unique())

    for vehicle_id in vehicle_ids:
        df_vehicle = df[df["Vehicle_ID"] == vehicle_id]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 트럭 외곽선
        ax.plot([0, truck_w, truck_w, 0, 0], [0, 0, truck_l, truck_l, 0], [0, 0, 0, 0, 0], color='black')
        ax.plot([0, truck_w, truck_w, 0, 0], [0, 0, truck_l, truck_l, 0], [truck_h, truck_h, truck_h, truck_h, truck_h], color='black')
        for x in [0, truck_w]:
            for y in [0, truck_l]:
                ax.plot([x, x], [y, y], [0, truck_h], color='black')

        # 박스 그리기
        for _, row in df_vehicle.iterrows():
            x, y, z = row["Lower_Left_X"], row["Lower_Left_Y"], row["Lower_Left_Z"]
            dx, dy, dz = row["Box_Width"], row["Box_Length"], row["Box_Height"]

            corners = [
                (x, y, z),
                (x + dx, y, z),
                (x + dx, y + dy, z),
                (x, y + dy, z),
                (x, y, z + dz),
                (x + dx, y, z + dz),
                (x + dx, y + dy, z + dz),
                (x, y + dy, z + dz)
            ]

            faces = [
                [corners[0], corners[1], corners[2], corners[3]],
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[1], corners[2], corners[6], corners[5]],
                [corners[4], corners[7], corners[3], corners[0]]
            ]

            ax.add_collection3d(Poly3DCollection(faces, alpha=0.4, edgecolor='k'))

        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")
        ax.set_xlim(0, truck_w)
        ax.set_ylim(0, truck_l)
        ax.set_zlim(0, truck_h)
        ax.set_title(f"차량 {vehicle_id} 적재 3D 시각화")
        plt.show()

# ===================== 실행 =====================
visualize_all_vehicles(df_valid)
