import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 데이터 로딩
df = pd.read_excel("C:/Contest/Cj/Result.xlsx")
df = df[df["Box_ID"].notna()]
df = df.sort_values(["Vehicle_ID", "Stacking_Order"])

# 특정 차량 선택
vehicle_id = 0  # 원하는 차량 번호
df_vehicle = df[df["Vehicle_ID"] == vehicle_id].reset_index(drop=True)

truck_w, truck_l, truck_h = 160, 280, 180

# 3D 시각화 설정
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 트럭 외곽선
ax.plot([0, truck_w, truck_w, 0, 0], [0, 0, truck_l, truck_l, 0], [0, 0, 0, 0, 0], color='black')
ax.plot([0, truck_w, truck_w, 0, 0], [0, 0, truck_l, truck_l, 0], [truck_h, truck_h, truck_h, truck_h, truck_h], color='black')
for x in [0, truck_w]:
    for y in [0, truck_l]:
        ax.plot([x, x], [y, y], [0, truck_h], color='black')

ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.set_xlim(0, truck_w)
ax.set_ylim(0, truck_l)
ax.set_zlim(0, truck_h)
ax.set_title(f"차량 {vehicle_id} 적재 3D 시각화")

# 상태 변수
box_index = [0]

# 박스 추가 함수
def draw_next(event):
    if box_index[0] >= len(df_vehicle):
        return

    row = df_vehicle.iloc[box_index[0]]
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
    plt.draw()
    box_index[0] += 1

# 버튼 생성
ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
btn = Button(ax_button, '다음 박스')
btn.on_clicked(draw_next)

plt.show()
