from math import gamma, sqrt

def h_linear_conflict(data, goal, size):
    """Linear Conflict + Manhattan Distance/Taxicab geometry"""
    distance = 0
    conflicts = 0
    N = size
    #for i in range(0, len(start)):
    for index in range(0, N * N):
        if data[index] and data[index] != goal[index]:
            x, y = index % N, index // N
            g_index = goal.index(data[index])
            g_x, g_y = g_index % N, g_index // N
            distance += abs(x - g_x) + abs(y - g_y)
            #distance += sqrt((x - g_x) * (x - g_x) + (y - g_y) * (y - g_y))
            a = data[index]
            b = goal[index]
            x1, y1 = data.index(b) % N, data.index(b) // N
            x2, y2 = goal.index(a) % N, goal.index(a) // N
            if (x == x1 or y == y1) and (x == x2 or y == y2):
                conflicts += 1
    
    print('Manathan distance ' + str(distance) + ' number of conflicts ' + str(conflicts))
    return distance + 2 * conflicts

def main():
    state = [
        4, 2, 5,
        1, 0, 6,
        3, 8, 7
    ]
    goal = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 0
    ]
    h = h_linear_conflict(data=state, goal=goal, size=3)
    print('linear conflicts : ' + str(h))

if __name__ == "__main__":
    main()