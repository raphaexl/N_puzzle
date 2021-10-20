#!/usr/bin/env python3
from os import PRIO_PGRP
import sys
import re

#Node or Cell or State 
class Node:
    def __init__(self, size, puzzle, g = 0, fcost = 0):
        """ Init node with the data, level of the node and calculate the heuristic fcost """
        self.size = size
        self.data = puzzle
        self.level = g
        self.fcost = fcost
        #self.h = heuristics()

    def generate_child(self):
        """ Generate child nodes from a given node by moving the blank space either in 4 dir {Up, down, right, left} """
        x, y = self.find(self.data, 0) #find the position of 0
        allowed_moves = [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]
        children = []
        # print(x, y, self.data)
        for i in allowed_moves:
            child = self.shuffle(self.data, x, y, i[0], i[1])
            if child is not None:
                child_node = Node(self.size, child, self.level + 1, 0)
                children.append(child_node)
        #sys.exit(0)
        return children
    
    def shuffle(self, puzz, x1, y1, x2, y2):
        if (x2 >= 0 and x2 < self.size) and (y2 >= 0 and y2 < self.size):
            temp_puz = []
            temp_puz = self.copy(puzz)
            temp = temp_puz[x1 + y1 * self.size]
            temp_puz[x1 + y1 * self.size] = temp_puz[x2 + y2 * self.size]
            temp_puz[x2 + y2 * self.size] = temp
            return temp_puz
        return None

    def copy(self, root):
        temp = []
        for i in root:
            temp.append(i)
        return temp
    
    def find(self, puz, x):
        for index in range(0, len(self.data)):
            if puz[index] == x:
                return  index % self.size, index // self.size
        return None

    def h_manathan(self):
        print('This is h-score with manathan distance')
        pass
    def h_misplaced(self):
        print('This h-score as the number of misplaced tiles by comparing the current state and the goal state or summation of the Manhattan distance between misplaced nodes.')
        pass
    def h_manathan(self):
        print('This is h-score with euclidean-distance')
        pass

    def heuristic(self):
        pass

class Puzzle:
    def __init__(self, size, start, goal):
        self.n = size
        self.start = start
        self.goal = goal
        self.open_list = []
        self.closed_list = []

    def __str__(self):
        return f'Puzzle{self.data}'

    def __repr__(self):
        return f'Puzzle(name={self.data})'
    
    def accept(self):
        puz = []
        for i in range(0, self.n):
            temp = input().split(' ')
            puz += [ int(x) for x in temp ]
        return puz

    def h(self, start, goal):
        """misplaced tiles heuristic"""
        temp = 0
        #for i in range(0, len(start)):
        for i in range(0, self.n * self.n):
            if start[i] != goal[i] and start[i] != 0:
                temp += 1
        return temp

    def f(self, currNode, goal):
        """ This is the cost function f(x) = g(x) + h(x) """
        return currNode.level + self.h(currNode.data, goal)  #level or g

    def print_separator(self, currentNode):
        print('')
        print('   |  ')
        print('   |  ')
        print("  \\\'/   ")
        for i, val in enumerate(currentNode.data):
            print(val, end=' ')
            if i == self.n - 1:
                print('')
            elif i > self.n:
                nextIndex = i + 1
                if nextIndex % (self.n) == 0:
                    print('')

    def solve(self):
        # print("Enter the start state matrix")
        # self.start = self.accept()
        # print("Enter the goal state matrix")
        # self.goal = self.accept()

        # self.start = [1, 2, 3,0, 4, 6,7, 5, 8]
        # self.goal = [1, 2, 3,4, 5, 6,7, 8, 0]
        """The actual algorithm"""
        start = Node(self.n, self.start, 0, 0)
        start.fcost = self.f(start, goal=self.goal)
        self.open_list.append(start) #add the start node to the open list
        print('\n\n')
        while True:
            currenNode = self.open_list[0] #pop the first element from the open list queue
            self.print_separator(currenNode)
            if  self.h(currenNode.data, self.goal) == 0: #break if we found the solution
                break
            for i in currenNode.generate_child():
                # print('start', i.data)
                # print('goal', self.goal)
                i.fcost = self.f(i, goal=self.goal)
                self.open_list.append(i)
            self.closed_list.append(currenNode) #already explored mark it as visited by puting it inside close list
            # sys.exit()
            del self.open_list[0]
            self.open_list.sort(key=lambda x: x.fcost, reverse=False)

class PuzzleParser:
    def __init__(self, fileName):
        self.size, self.start_state = self.parsePuzzle(fileName)
        self.goal_state = self.computeGoalState()

    def parsePuzzle(self, fileName):
        lines = []
        with open(fileName) as f:
            lines = f.readlines()
        count = 0
        size = 0
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
                if (size == 0 and len(s) == 1):
                    if (count == 0):
                        size = int(s)
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
        grid = arr.copy()
        grid.remove(0)
        for i in range(1, len(grid)):
            for j in range(i - 1, 0, -1):
                if (grid[j] <= grid[j + 1]):
                    break
                grid[j + 1], grid[j] = grid[j], grid[j + 1]
                inv_count += 1
        return inv_count
    
    def find_blank_position(self, puzzle):
        N = self.size
        blank_row = (len(puzzle) - 1 - puzzle.index(0)) // N
        return blank_row
        for i in range(N - 1, 0, -1):
            for j in range(N - 1, 0, -1):
                if (puzzle[i * N + j] == 0):
                    return N - i
        return -1

    def parity(self, puzzle):
        inv_count = self.get_inversion_count(puzzle)
        N = self.size
        if (N & 1):
            print(f"inversion count: {inv_count}")
            return (inv_count & 1)
        else:
            pos = self.find_blank_position(puzzle)
            print(f"inversion count {inv_count} blank row positon: {pos}")

            return (inv_count + pos) & 1
            if (pos & 1):
                return (inv_count & 1)
            else:
                return not inv_count & 1
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
        return False
        # puzzle = self.goal_state
        # goalParity = self.parity(puzzle)
        # print('')
        # print('Goal state is pair ? ' + 'Yes' if goalParity else 'No')
        # print('Goal is ',puzzle)
        # return startParity == goalParity


def main(argv):
    if (len(argv) != 1):
        print('Usage: ./n-puzzle  npuzzle-start-state.txt')
        sys.exit(0)
    puzzleParser = PuzzleParser(argv[0])
    if not puzzleParser.solvable():
        print('Sorry the puzzle provided is not solvable :-(')
        sys.exit(0)
    # puz = Puzzle(puzzleParser.size, puzzleParser.get_start_state(), puzzleParser.get_goal_state())
    # puz.solve()
    sys.exit(0)

 
if __name__ == "__main__":
    main(sys.argv[1:])
else:
    print ("Executed when imported")