from ortools.sat.python import cp_model

def ip(b):
    model = cp_model.CpModel()
    # 변수 정의: 정수 변수 10개, 각 변수는 0~10
    x = [model.NewIntVar(0, 10, f'x{i}') for i in range(11)]
    capacity_matrix = [[84 , 84 , 0],
                    [168 , 42 , 0],
                    [42 , 105 , 0],
                    [0 , 84 , 15 ],
                    [0 , 42 , 30],
                    [84 , 0 , 30],
                    [168 , 0 , 15],
                    [ 210 , 0 , 0],
                    [ 0 , 126 , 0],
                    [ 0 , 0 , 45],
                    [84, 42 , 15]                
    ]

    # 제약식 추가
    for i in range(3):
        model.Add(sum(capacity_matrix[j][i] * x[j] for j in range(len(x))) >= b[i])


    model.Minimize(sum(x))

    # 솔버 실행
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # 결과 출력
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(11):
            print(f"x{i} = {solver.Value(x[i])}")
        print("최적값 =", solver.ObjectiveValue())
    else:
        print("해를 찾을 수 없습니다.")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = [solver.Value(var) for var in x]
        return solution
    else:
        return "FUCK 정수계획법 실패"

sol = ip({0 :151 ,1:161 , 2:125})
print(sol)