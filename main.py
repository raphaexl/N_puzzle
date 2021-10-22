#!/usr/bin/env python3
from os import pread
import sys
import re
from heapq import heappop, heappush
from math import  sqrt

class Node:
    def __init__(self, parent, puzzle, g = 0, fcost = 0, move_dir = -1):
        """ Init node with the data, level of the node and calculate the heuristic fcost """
        self.parent = parent
        self.data = puzzle
        self.level = g
        self.fcost = fcost
        self.move = move_dir

    def __lt__(self, other):
        return  self.fcost < other.fcost

    def length(self):
        if (self.parent == None):
            return 0
        return 1 + self.parent.length()

class Puzzle:
    def __init__(self, size, start, goal, heuristic=0):
        self.n = size
        self.start = start
        self.goal = goal
        self.h = self.setup_heuristic(heuristic)
        self.complexityInTime = 0
        self.complexityInSize = 0
        self.move_count = 0

    def __str__(self):
        return f'Puzzle{self.data}'

    def __repr__(self):
        return f'Puzzle(name={self.data})'

    def h_manathan_distance(self, state, goal):
        """Manhattan Distance/Taxicab geometry"""
        distance = 0
        N = self.n
        for index in range(0, N * N):
            if state[index] and state[index] != goal[index]:
                x, y = index % N, index // N
                g_index = goal.index(state[index])
                g_x, g_y = g_index % N, g_index // N
                distance += abs(x - g_x) + abs(y - g_y)
        return distance
    

    def h_misplaced_tiles(self, state, goal):
        """Hamming Distance/Misplaced Tiles"""
        distance = 0
        N = self.n
        for i in range(0, N * N):
            if state[i] and state[i] != goal[i]:
                distance += 1
        return distance
        
    def h_linear_conflict(self, state, goal):
        """Linear Conflict + Manhattan Distance/Taxicab geometry"""
        distance = 0
        conflicts = 0
        N = self.n
        for index in range(0, N * N):
            if state[index] and state[index] != goal[index]:
                x, y = index % N, index // N
                g_index = goal.index(state[index])
                g_x, g_y = g_index % N, g_index // N
                distance += abs(x - g_x) + abs(y - g_y)
                a = state[index]
                b = goal[index]
                x1, y1 = state.index(b) % N, state.index(b) // N
                x2, y2 = goal.index(a) % N, goal.index(a) // N
                if (x == x1 or y == y1) and (x == x2 or y == y2):
                    conflicts += 1
        return distance + 2 * conflicts

    def h_euclidean_distance(self, state, goal):
        """Euclidean"""
        distance = 0
        N = self.n
        for index in range(0, N * N):
            if state[index] and state[index] != goal[index]:
                x, y = index % N, index // N
                g_index = goal.index(state[index])
                g_x, g_y = g_index % N, g_index // N
                distance += sqrt((x - g_x) * (x - g_x) + (y - g_y) * (y - g_y))
        return distance

    def greedy(self, state, goal):
        return 0

    def setup_heuristic(self, user_heuristic):
        """Choose the appropriate heuristic and calculate using it 3 is greedy 0"""
        if (user_heuristic == 0):
            return self.greedy
        elif user_heuristic == 1:
            return self.h_manathan_distance
        elif user_heuristic == 2:
            return self.h_misplaced_tiles
        elif user_heuristic == 3:
            return  self.h_linear_conflict
        elif user_heuristic == 4:
            return self.h_euclidean_distance
        return self.h_euclidean_distance
    
    def generate_child(self, puzz):
        """ Generate child nodes from a given node by moving the blank space either in 4 dir {Up, down, right, left} """
        x, y = self.find(puzz, 0) #find the position of 0
        allowed_moves = [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]
        children = []
        append = children.append
        i = 0
        while i < 4:
            child = self.shuffle(puzz, x, y, allowed_moves[i][0], allowed_moves[i][1])
            if child is not None:
                child_node = {"dir" : i, "data" : child}
                append(child_node)
            i += 1
        return children
        for i, move in enumerate(allowed_moves):
            child = self.shuffle(node.data, x, y, move[0], move[1])
            if child is not None:
                child_node = {"dir" : i, "data" : child}
                children.append(child_node)
        return children
    
    def shuffle(self, puzz, x1, y1, x2, y2):
        N = self.n
        if (x2 >= 0 and x2 < N) and (y2 >= 0 and y2 < N):
            temp_puz = []
            temp_puz = puzz[:] #Copy
            index1 = x1 + y1 * N
            index2 = x2 + y2 * N
            temp_puz[index1], temp_puz[index2] = temp_puz[index2], temp_puz[index1] #copy
            return temp_puz
        return None
    
    def find(self, puz, x):
        index = puz.index(x)
        N = self.n
        return index % N, index // N
        if index >= 0:
            return index % N, index // N
        return None

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

    def display_solution(self, endNode):
        print('---Solution---')
        required_moves = endNode.length()
        print('Complexity in time : ', (self.complexityInTime))
        print('Complexity in size : ', self.complexityInSize)
        print('Required moves : ' + str(required_moves))
        self.print_path(endNode)

    def solve(self):
        #setattr(ListNode, "__lt__", lambda self, other: self.val <= other.val)
        if  self.start == self.goal: #break if we found the solution
            print('The Provided State is the Solution...')
            self.print_path(Node(None, self.n, self.start))
            return 
        """The actual algorithm"""
        open_list = []
        closed_list = set()
        start = Node(None, self.start, 0, self.h(self.start, self.goal)) #same as f = h cause g is zero at start
        heappush(open_list, start) #add the start node to the open list
        #closed_list.add(str(start.data))
        import time
        start_time = time.time()
        while open_list:
            currentNode = heappop(open_list) #pop the first element from the open list queue
            self.complexityInTime += 1
            if  currentNode.data == self.goal: #break if we found the solution
                print("--- %s seconds ---" % (time.time() - start_time))
                #self.display_solution(currentNode)
                return
            for child in self.generate_child(currentNode.data):
                g = currentNode.level + 1
                h = self.h(currentNode.data, self.goal)
                fcost = g + h
                if not (str(child["data"]) in closed_list):
                    child_node = Node(currentNode,  child["data"], g, fcost, child['dir'])
                    heappush(open_list, child_node)
                    self.complexityInSize += 1
            closed_list.add(str(currentNode.data)) #already explored mark it as visited by puting it inside close list
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
        try:
            with open(fileName) as f:
                lines = f.readlines()
        except IOError as e:
            print(e)
            sys.exit(1)
        except:
            print('Error loading provided file :-(')
            sys.exit(1)
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
        return size, data
    
    def computeGoalState(self):
        print('size ',self.size)
        m = self.size
        n = self.size
        arr = [[None]*n for _ in range(m)]
        k = 0; l = 0
        currNumber = 0
        while (k < m and l < n) :
            for i in range(l, n) : 
                currNumber += 1
                arr[k][i] = currNumber
            k += 1
            for i in range(k, m) : 
                currNumber += 1
                arr[i][n - 1] = currNumber
            n -= 1
            if ( k < m) : 
                
                for i in range(n - 1, (l - 1), -1) : 
                    currNumber += 1
                    arr[m - 1][i] = currNumber
                m -= 1
            if (l < n) : 
                for i in range(m - 1, k - 1, -1) : 
                    currNumber += 1
                    arr[i][l] = currNumber
                l += 1
        
        state = [j for sub in arr for j in sub]
        state[state.index(self.size * self.size)] = 0
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
        return -1

    def parity(self, puzzle):
        inv_count = self.get_inversion_count(puzzle)
        N = self.size
        if (N & 1):
            return (inv_count & 1)
        else:
            pos = self.find_blank_position(puzzle)
            if (pos & 1):
                return not (inv_count & 1)
            else:
                return inv_count & 1

        
    def solvable(self):
        puzzle = self.start_state
        startParity = self.parity(puzzle)
        print('Start is ',puzzle)
        puzzle = self.goal_state
        goalParity = self.parity(puzzle)
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
        print("Wrong parameters :-( check you arguments")