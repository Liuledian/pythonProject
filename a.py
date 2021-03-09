def solution(K, A):
    N = len(A)
    M = len(A[0])
    houses = []
    for i in range(N):
        for j in range(M):
            if A[i][j] == 1:
                houses.append((i, j))

    result = 0
    for i in range(N):
        for j in range(M):
            flag = True
            for house in houses:
                distance = abs(house[0] - i) + abs(house[1] - j)
                if distance == 0 or distance > K:
                    flag = False
                    break
            if flag:
                result += 1
    return result


a = [[0,0,0,0],
     [0,0,1,0],
     [1,0,0,1]]

b= [[0,1],
    [0,0]]

c= [[0,0,0,1],
    [0,1,0,0],
    [0,0,1,0],
    [1,0,0,0],
    [0,0,0,0]]

k1 = 2
k2 = 1
k3 = 4

print(solution(k1, a))
print(solution(k2, b))
print(solution(k3, c))
