from itertools import permutations

edges = []
for i in range(1, 8):
    edges.append((i,i+1))
edges.extend([(8,1),(1,5),(2,6),(3,7),(4,8)])
print(len(edges),edges)


def compute_dilation(circle):
    c = len(circle)
    dilation = 0
    for edge in edges:
        si = circle.index(edge[0])
        ti = circle.index(edge[1])
        dis = abs(si-ti)
        dis = min(dis, c - dis)
        dilation = max(dilation, dis)
    return dilation


def min_dialation():
    perms = permutations(list(range(2,9)))
    n = 0
    min_dila = 8
    for perm in perms:
        n += 1
        dila = compute_dilation((1,) + perm)
        if dila < min_dila:
            min_dila = dila
    print("number of permus", n)
    return min_dila


if __name__ == '__main__':
    print(min_dialation())