#!/usr/bin/env python3
import sys
import re
from heapq import heappop, heappush
from math import  sqrt


#Node or Cell or State 
# A class for Priority Queue
class priorityQueue:
     
    # Constructor to initialize a
    # Priority Queue
    def __init__(self):
        self.heap = []
 
    # Inserts a new key 'k'
    def push(self, k):
        heappush(self.heap, k)
 
    # Method to remove minimum element
    # from Priority Queue
    def pop(self):
        return heappop(self.heap)
 
    # Method to know if the Queue is empty
    def empty(self):
        if not self.heap:
            return True
        else:
            return False
class Node:
    def __init__(self, parent, size, puzzle, g = 0, fcost = 0, selected_h = 1, move_dir = -1):
        """ Init node with the data, level of the node and calculate the heuristic fcost """
        self.parent = parent
        self.size = size
        self.data = puzzle
        self.level = g
        self.fcost = fcost
        self.move = move_dir
        self.heuristic = selected_h #self.setup_heuristic(selected_h)

    def h_manathan_distance(self, goal):
        """Manhattan Distance/Taxicab geometry"""
        distance = 0
        N = self.size
        #for i in range(0, len(start)):
        for index in range(0, N * N):
            if self.data[index] and self.data[index] != goal[index]:
                x, y = index % N, index // N
                g_index = goal.index(self.data[index])
                g_x, g_y = g_index % N, g_index // N
                distance += abs(x - g_x) + abs(y - g_y)
        return distance
    

    def h_misplaced_tiles(self, goal):
        """Hamming Distance/Misplaced Tiles"""
        distance = 0
        N = self.size
        #for i in range(0, len(start)):
        for i in range(0, N * N):
            if self.data[i] and self.data[i] != goal[i]:
                distance += 1
        return distance
        
    def h_linear_conflict(self, goal):
        """Linear Conflict + Manhattan Distance/Taxicab geometry"""
        distance = 0
        conflicts = 0
        N = self.size
        #for i in range(0, len(start)):
        for index in range(0, N * N):
            if self.data[index] and self.data[index] != goal[index]:
                x, y = index % N, index // N
                distance += sqrt(x * x + y * y)
                a = self.data[index]
                b = goal[index]
                x1, y1 = self.data.index(b) % N, self.data.index(b) // N
                x2, y2 = goal.index(a) % N, goal.index(a) // N
                if (x == x1 or y == y1) and (x == x2 or y == y2):
                    conflicts += 1
        return distance + 2 * conflicts

    def h_euclidean_distance(self, goal):
        """Euclidean"""
        distance = 0
        N = self.size
        for index in range(0, N * N):
            if self.data[index] and self.data[index] != goal[index]:
                x, y = index % N, index // N
                g_index = goal.index(self.data[index])
                g_x, g_y = g_index % N, g_index // N
                distance += sqrt((x - g_x) * (x - g_x) + (y - g_y) * (y - g_y))
        return distance
    
    def greedy(seelf, goal):
        return 0

    def setup_heuristic(self, heuristic):
        """Choose the appropriate heuristic and calculate using it 3 is greedy 0"""
        if (heuristic == 0):
            return self.greedy
        elif heuristic == 1:
            return self.h_manathan_distance
        elif heuristic == 2:
            return self.h_misplaced_tiles
        elif heuristic == 3:
            return  self.h_linear_conflict
        elif heuristic == 4:
            return self.h_euclidean_distance
        print('Heiristic unspecified for this Node Euclidiane distance :-(')
        return self.h_euclidean_distance

    def h(self, goal):
        """Choose the appropriate heuristic and calculate using it 3 is greedy 0"""
        if (self.heuristic == 0):
            return 0
        elif self.heuristic == 1:
            return self.h_manathan_distance(goal)
        elif self.heuristic == 2:
            return self.h_misplaced_tiles(goal)
        elif self.heuristic == 3:
            return  self.h_linear_conflict(goal)
        elif self.heuristic == 4:
            return self.h_euclidean_distance(goal)
        return self.h_euclidean_distance(goal)

    def __eq__(self,other):
        #return self.fcost == other.fcost
        return self and other and self.fcost == other.fcost
    
    def __lt__(self, other):
        #Some optimization for g and h is it necessary
        return self.fcost < other.fcost
        return self and other and  self.fcost <= other.fcost

class Puzzle:
    def __init__(self, size, start, goal, heuristic=0):
        self.n = size
        self.start = start
        self.goal = goal
        self.open_list = []
        self.closed_list = set()
        self.heuristic = heuristic

    def __str__(self):
        return f'Puzzle{self.data}'

    def __repr__(self):
        return f'Puzzle(name={self.data})'
    
    def generate_child(self, node):
        """ Generate child nodes from a given node by moving the blank space either in 4 dir {Up, down, right, left} """
        x, y = self.find(node.data, 0) #find the position of 0
        allowed_moves = [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]
        children = []
        # print(x, y, self.data)
        for i, move in enumerate(allowed_moves):
            child = self.shuffle(node.data, x, y, move[0], move[1])
            if child is not None:
                child_node = {"dir" : i, "data" : child} # Node(node, self.n, child, node.level + 1, 0, self.heuristic, i)
                children.append(child_node)
        #sys.exit(0)
        return children
    
    def shuffle(self, puzz, x1, y1, x2, y2):
        N = self.n
        if (x2 >= 0 and x2 < N) and (y2 >= 0 and y2 < N):
            temp_puz = []
            temp_puz = puzz[:] #Copy
            # temp = temp_puz[x1 + y1 * self.size]
            # temp_puz[x1 + y1 * self.size] = temp_puz[x2 + y2 * self.size]
            # temp_puz[x2 + y2 * self.size] = temp
            # return temp_puz
            #swap
            index1 = x1 + y1 * N
            index2 = x2 + y2 * N
            temp_puz[index1], temp_puz[index2] = temp_puz[index2], temp_puz[index1]
            return temp_puz
        return None
    
    def find(self, puz, x):
        index = puz.index(x)
        N = self.n
        if index >= 0:
            return index % N, index // N
        return None
        for index in range(0, len(self.data)):
            if puz[index] == x:
                return  index % self.size, index // self.size
        return None
    def f(self, currNode, goal):
        """ This is the cost function f(x) = g(x) + h(x) """
        return currNode.level + currNode.h(goal)  #level or g

    def print_separator(self, currentNode):
        s_str = ''
        if currentNode.move > -1:
            dir_set = {
                0: "Down",
                1: "Up",
                2: "Right",
                3: "Left"
            }
            s_str += dir_set.get(currentNode.move) + '\n'
        s_bar = '-\t' * (2 + self.n)
        s_str += s_bar + '\n' + '|\t'
        for i, val in enumerate(currentNode.data):
            s_str = s_str + str(val) + '\t'
            if i == self.n - 1:
                s_str = s_str  + '|\n'
                s_str += '|\t'
            elif i > self.n:
                nextIndex = i + 1
                if nextIndex % (self.n) == 0:
                    s_str = s_str  + '|'
                    if (nextIndex < self.n * self.n):
                        s_str += '\n|\t'
        s_str += '\n' + s_bar
        print(s_str)

    def solve_1(self):
        #setattr(ListNode, "__lt__", lambda self, other: self.val <= other.val)
        startNode = Node(None, self.n, self.start, 0, 0, self.heuristic)
        startNode.fcost = self.f(startNode, self.goal)
        goalNode = Node(None, self.n, self.goal, 0, 0, self.heuristic)
        opened = []
        closed = []
        heappush(opened, startNode)
        while opened:
            process = heappop(opened)
            if (process.data == goalNode.data):
                self.print_path(process)
                break
            closed.append(process)
            for node in process.generate_child():
                node.fcost = self.f(node, goal=self.goal)
                if node in closed:
                    continue
                if not node in opened:
                    heappush(opened, node)
                else:
                    index = opened.index(node)
                    currentNode = opened[index] #(x for x in opened if x == node)
                    if node.level < currentNode.level:
                        currentNode.level = node.level
                        currentNode.fcost = node.fcost
                        currentNode.parent = node.parent
        print('Not possible to reach goal')

    def solve(self):
        # print("Enter the start state matrix")
        # self.start = self.accept()
        # print("Enter the goal state matrix")
        # self.goal = self.accept()

        # self.start = [1, 2, 3,0, 4, 6,7, 5, 8]
        # self.goal = [1, 2, 3,4, 5, 6,7, 8, 0]
        if  self.start == self.goal: #break if we found the solution
            print('The Provided State is the Solution...')
            self.print_path(Node(None, self.n, self.start))
            return 
        """The actual algorithm"""
        start = Node(None, self.n, self.start, 0, 0, self.heuristic)
        start.fcost = self.f(start, goal=self.goal) #same as h cause g is zero at start
        heappush(self.open_list, start) #add the start node to the open list
        self.closed_list.add(str(start.data))
        while self.open_list:
            currentNode = heappop(self.open_list) #pop the first element from the open list queue
            #self.print_separator(currentNode)
            if  currentNode.data == self.goal: #break if we found the solution
                print('Solution')
                self.print_path(currentNode)
                return
            for child in self.generate_child(currentNode):
                # print('start', i.data)
                # print('goal', self.goal)
                g = currentNode.level
                h = currentNode.h(self.goal)
                fcost = g + h
                if (str(child["data"]) not in self.closed_list):
                    child_node = Node(currentNode, self.n, child["data"], currentNode.level + 1, fcost, self.heuristic, child['dir'])
                    heappush(self.open_list, child_node)
            self.closed_list.add(str(currentNode.data)) #already explored mark it as visited by puting it inside close list
            # sys.exit()
            # del self.open_list[0]
            # self.open_list.sort(key=lambda x: x.fcost, reverse=False)
        print('Error : No path found !')
    def print_path(self, root):
        if root == None:
            return
        self.print_path(root.parent)
        self.print_separator(root)
        print()

class PuzzleParser:
    def __init__(self, fileName):
        self.size, self.start_state = self.parsePuzzle(fileName)
        self.goal_state = self.computeGoalState()

    def parsePuzzle(self, fileName):
        lines = []
        with open(fileName) as f:
            lines = f.readlines()
        count = 0
        size = -1
        data = []
        for line in lines:
            s = line.strip()
            if (not s or s[0] == '#'):#may be only #
                continue
            s = s[:s.index('#')] if '#' in s and s.index('#') else s
            if not re.match("^ *[0-9][0-9 ]*$", s):
                print('Error not allowed characters, line: ' +line, end='')
                sys.exit(0)
            else:                
                if (size == -1 and s.isnumeric()):
                    if (count == 0):
                        size = int(s)
                        if (size < 3):
                            print('Error the puzzle must be greater of equal to 3')
                            sys.exit(0)
                        continue
                    else:
                        print('Error the size not must be first specified be fore anything')
                        sys.exit(0)
                s_list = s.split()
                if (len(s_list) != size):
                    print('Error a line that has not the specified size, line: ', line, end='')
                    sys.exit(0)
                i_list = [int(x) for x in s_list]
                for x in i_list:
                    if x < 0 or x >= size * size:
                        print('Error a number not in the range of the allowed numbers, line: ', line, end='')
                        sys.exit(0)  
                data += i_list
                count += 1
        if (size == -1):
            print('Error could not read the puzzle dimension')
            sys.exit(0)
        if (count != size):
            print('There is a mismatch between the specified size and the provided map size')
            sys.exit(0)
        print('Parsing done successfully :-) ')
        print('data', data)
        return size, data
    


    def computeGoalState(self):
        print('size ',self.size)
        m = self.size
        n = self.size

        arr = [[None]*n for _ in range(m)]
        k = 0; l = 0
  
        ''' k - starting row index 
            m - ending row index 
            l - starting column index 
            n - ending column index 
            i - iterator '''

        currNumber = 0
        while (k < m and l < n) : 
            
            # Compute the first row from 
            # the remaining rows  
            for i in range(l, n) : 
                currNumber += 1
                arr[k][i] = currNumber
            k += 1
    
            # Compute the last column from 
            # the remaining columns  
            for i in range(k, m) : 
                # print(a[i][n - 1], end = " ") 
                currNumber += 1
                arr[i][n - 1] = currNumber
            n -= 1
            # Compute the last row from 
            # the remaining rows  
            if ( k < m) : 
                
                for i in range(n - 1, (l - 1), -1) : 
                    # print(a[m - 1][i], end = " ") 
                    currNumber += 1
                    arr[m - 1][i] = currNumber
                m -= 1
            # Compute the first column from 
            # the remaining columns  
            if (l < n) : 
                for i in range(m - 1, k - 1, -1) : 
                    currNumber += 1
                    arr[i][l] = currNumber
                l += 1
        
        state = [j for sub in arr for j in sub]
        state[state.index(self.size * self.size)] = 0
   #     state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        print('\n target state ', state)
        print(' => arr', arr)
        return state


    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state
    
    def get_size(self):
        return self.size

    def get_inversion_count(self, arr):
        inv_count = 0
        N = self.size
        # grid = arr.copy()
        # grid.remove(0)
        # for i in range(1, len(grid)):
        #     for j in range(i - 1, 0, -1):
        #         if (grid[j] <= grid[j + 1]):
        #             break
        #         grid[j + 1], grid[j] = grid[j], grid[j + 1]
        #         inv_count += 1
        # return inv_count
        for i in range(N * N - 1):
            for j in range(i + 1, N * N):
                if arr[j] and arr[i] and arr[i] > arr[j]:
                    inv_count += 1
        return inv_count
    
    def find_blank_position(self, puzzle):
        N = self.size
        for i in range(N - 1, -1, -1):
            for j in range(N - 1, -1, -1):
                if (puzzle[i * N + j] == 0):
                    return N - i
        print('What ??? N = ', N)
        print('(3, 0)', puzzle[3 * N + 0])
        return -1

    def parity(self, puzzle):
        inv_count = self.get_inversion_count(puzzle)
        N = self.size
        if (N & 1):
            print(f"inversion count: {inv_count}")
            return (inv_count & 1)
        else:
            pos = self.find_blank_position(puzzle)
            print(f"inversion count: {inv_count} position from bottom : {pos}")
            # return (inv_count + pos) & 1
            if (pos & 1):
                return not (inv_count & 1)
            else:
                return inv_count & 1
    # def check_parity(self, puzzle):

        
    def solvable(self):
        # y_puzzle = [12, 1, 10, 2,
        # 7, 11, 4, 14,
        # 5, 0, 9, 15,
        # 8, 13, 6, 3]

        y_1puzzle = [
            6, 13, 7, 10,
                    8, 9, 11, 0,
                    15, 2, 12, 5,
                    14, 3, 1, 4
        ]
        y_2puzzle = [
             13, 2, 10, 3,
                    1, 12, 8, 4,
                    5, 0, 9, 6,
                    15, 14, 11, 7
        ]
        n_puzzle = [
            3, 9, 1, 15,
            14, 11, 4, 6,
                    13, 0, 10, 12,
                    2, 7, 8, 5
        ]
        # self.size = 4
        puzzle = self.start_state
        startParity = self.parity(puzzle)
        print('Start state is pair ? ' + 'Yes' if startParity else 'No')
        puzzle = self.goal_state
        goalParity = self.parity(puzzle)
        print('Goal state is pair ? ' + 'Yes' if goalParity else 'No')
        print('Goal is ',puzzle)
        return startParity and goalParity


def main(fileName, heuristic):
    puzzleParser = PuzzleParser(fileName)
    if not puzzleParser.solvable():
        print('Sorry the puzzle provided is not solvable :-(')
        sys.exit(0)
    print('The puzzle is solvable')
    puz = Puzzle(puzzleParser.size, puzzleParser.get_start_state(), puzzleParser.get_goal_state(), heuristic)
    puz.solve()
    sys.exit(0)

 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Solve a puzzle of N * N dimensions', formatter_class= argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--file', help='filename for the puzzle start state', dest="fileName", required=True)
    h_help = '1 Manhattan Distance/Taxicab geometry as heuristic [default]\n'  + '2 Hamming Distance/Misplaced Tiles as heuristic\n'+'3 Linear Conflict + Manhattan Distance/Taxicab geometry as heuristic (Recommended)\n'+'4 Euclidian Distance as heuristic\n'+'0 Greedy search no heuristic'
    parser.add_argument('--heuristic', help=h_help, type=int, default=1)
    options = parser.parse_args()
    if options.fileName and options.heuristic >=  0:
        main(options.fileName, options.heuristic)
else:
    print ("Executed when imported")