

class Node:
    def __init__(self, data, level, fval):
        """Init node with the data, level of the node and calculate the heuristic fvalue"""
        self.data = data
        self.level = level
        self.fval = fval
    
    def generate_child(self):
        """Generate child nodes from a given node by moving the blank space either in 4 dir {Up, down, right, left} """
        x, y = self.find(self.data, '_')
        val_list = [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]
        children = []
        print(x, y, self.data)
        for i in val_list:
            child = self.shuffle(self.data, x, y, i[0], i[1])
            if child is not None:
                print('called for ', child)
                child_node = Node(child, self.level + 1, 0)
                children.append(child_node)

        return children

    def shuffle(self, puz, x1, y1, x2, y2):
        if (x2 >= 0 and x2 < 3 and y2 >= 0 and y2 < 3):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None
    
    def copy(self, root):
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp
    
    def find(self, puz, x):
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data)):
                if puz[i][j] == x:
                    return i, j
    
class Puzzle:
    def __init__(self, size):
        self.n = size
        self.open = []
        self.closed = []

    def accept(self):
        puz = []
        for i in range(0, self.n):
            temp = input().split(' ')
            puz.append(temp)
        return puz
    
    def f(self, start, goal):
        return self.h(start.data, goal) + start.level
    
    def h(self, start, goal):
        temp = 0
        for i in range(0, self.n):
            for j in range(0, self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp

    def process(self):
        # print("Enter the start state matrix")
        # start = self.accept()
        # print("Enter the goal state matrix")
        # goal = self.accept()
        start = [['1', '2', '3'],['_', '4', '6'],['7', '5', '8']]
        goal = [['1', '2', '3'],['8', '_', '4'],['7', '6', '5']]
        start = Node(start, 0, 0)
        start.fval = self.f(start, goal)
        self.open.append(start) #put the start node in the open list
        print('\n\n')
        while True:
            curr = self.open[0]
            print('')
            print('   |  ')
            print('   |  ')
            print("  \\\'/   ")
            
            """We've reached the goal if the current state  and goal state is 0"""
            if self.h(curr.data, goal) == 0:
                break
            for i in curr.generate_child():
                i.fval = self.f(i, goal)
                self.open.append(i)
            

            self.closed.append(curr)
            del self.open[0]
            #sort the open list based on the heuristic f value
            self.open.sort(key = lambda x: x.fval , reverse = False)

if __name__ == "__main__":
    puz = Puzzle(3)
    puz.process()