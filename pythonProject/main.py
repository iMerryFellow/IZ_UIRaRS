import random
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

# Создаем матрицу 22 на 22 и рандомно заполняем
matrix = np.zeros((22, 22), dtype=np.uint8)
fill_coef = 0.4
num_elements = int(fill_coef * matrix.size)
nonzero_indices = np.random.choice(matrix.size, num_elements, replace=False)
matrix[np.unravel_index(nonzero_indices, matrix.shape)] = 255


def add_edges(matrix):      # Создание границ лабиринта
    rows, cols = len(matrix), len(matrix[0])

    new_rows = rows + 2
    new_cols = cols + 2
    new_matrix = [[0] * new_cols for i in range(new_rows)]

    for i in range(rows):
        for j in range(cols):
            new_matrix[i + 1][j + 1] = matrix[i][j]

    for j in range(1, new_cols - 1):
        new_matrix[0][j] = 0
        new_matrix[new_rows - 1][j] = 0

    for i in range(1, new_rows - 1):
        new_matrix[i][0] = 0
        new_matrix[i][new_cols - 1] = 0

    new_matrix[0][0] = 0
    new_matrix[0][new_cols - 1] = 0
    new_matrix[new_rows - 1][0] = 0
    new_matrix[new_rows - 1][new_cols - 1] = 0

    return new_matrix


new_matrix = add_edges(matrix)

# Создаем чб картинку из матрицы
matrix = np.asarray(new_matrix, dtype=np.uint8)
img = Image.fromarray(matrix).convert('RGB')
img.save('maze_0.png')

# Матрица местоположения агентов
matrix_agent = np.zeros((3, 2), dtype=np.uint8)
agent = 0

# Расскидываем агентов по полю
while agent != 3:
    row = random.randint(0, 21)
    col = random.randint(0, 21)
    if matrix[row][col] == 255:
        matrix[row][col] = 120
        matrix_agent[agent][0] = row
        matrix_agent[agent][1] = col
        agent += 1

img = Image.fromarray(matrix).convert('RGB')
img.save('maze_with_agents.png')

for row in matrix:
    print(row)
print('\n')

for row in matrix_agent:
    print(row)
print('\n')

# Создаем базу ключей местоположения
tik = []
# Начальный ключ
tik.append((matrix_agent[0][0])*1000000000000 + (matrix_agent[0][1])*10000000000 + (matrix_agent[1][0])*10000000 + (matrix_agent[1][1])*100000 + (matrix_agent[2][0])*100 + (matrix_agent[2][1]))

# Создаем ключ начального местоположения
graph = {}


while len(tik) > 0:
    variations = []
    r1_move = [0]
    r2_move = [0]
    r3_move = [0]
    x1 = tik[0]//1000000000000
    y1 = tik[0]%1000000000000//10000000000
    x2 = tik[0]%10000000000//10000000
    y2 = tik[0]%10000000//100000
    x3 = tik[0]%100000//100
    y3 = tik[0]%100//1

    # Проверка возможности передвижение каждого из агентов
    if matrix[x1+1 -1][y1+1] == 255:
        r1_move.append(4)                 #L 4
    if matrix[x1+1 +1][y1+1] == 255:
        r1_move.append(2)                 #R 2
    if matrix[x1+1][y1+1 -1] == 255:
        r1_move.append(1)                 #U 1
    if matrix[x1+1][y1+1 +1] == 255:
        r1_move.append(3)                 #D 3

    if matrix[x2+1 -1][y2+1] == 255:
        r2_move.append(4)                 #L 4
    if matrix[x2+1 +1][y2+1] == 255:
        r2_move.append(2)                 #R 2
    if matrix[x2+1][y2+1 -1] == 255:
        r2_move.append(1)                 #U 1
    if matrix[x2+1][y2+1 +1] == 255:
        r2_move.append(3)                 #D 3

    if matrix[x3+1 -1][y3+1] == 255:
        r3_move.append(4)                 #L 4
    if matrix[x3+1 +1][y3+1] == 255:
        r3_move.append(2)                 #R 2
    if matrix[x3+1][y3+1 -1] == 255:
        r3_move.append(1)                 #U 1
    if matrix[x3+1][y3+1 +1] == 255:
        r3_move.append(3)                 #D 3

    # Записываем все возможные вариации перемещений
    for a1 in range(len(r1_move)):
        for a2 in range(len(r2_move)):
            for a3 in range(len(r3_move)):
                variations.append([r1_move[a1],r2_move[a2],r3_move[a3]])

    # Очищаем начальную вариацию
    del variations[0]
    log = []
    counter = 0

    # Примитив передвижений, в зависимости от возможностей
    for n in range(len(variations)):
        if variations[n][0] == 0:
            x1_otn, y1_otn = 0, 0
        elif variations[n][0] == 1:
            x1_otn, y1_otn = 0, -1
        elif variations[n][0] == 2:
            x1_otn, y1_otn = 1, 0
        elif variations[n][0] == 3:
            x1_otn, y1_otn = 0, 1
        elif variations[n][0] == 4:
            x1_otn, y1_otn = -1, 0

        if variations[n][1] == 0:
            x2_otn, y2_otn = 0, 0
        elif variations[n][1] == 1:
            x2_otn, y2_otn = 0, -1
        elif variations[n][1] == 2:
            x2_otn, y2_otn = 1, 0
        elif variations[n][1] == 3:
            x2_otn, y2_otn = 0, 1
        elif variations[n][1] == 4:
            x2_otn, y2_otn = -1, 0

        if variations[n][2] == 0:
            x3_otn, y3_otn = 0, 0
        elif variations[n][2] == 1:
            x3_otn, y3_otn = 0, -1
        elif variations[n][2] == 2:
            x3_otn, y3_otn = 1, 0
        elif variations[n][2] == 3:
            x3_otn, y3_otn = 0, 1
        elif variations[n][2] == 4:
            x3_otn, y3_otn = -1, 0

        # Координаты после перемещения
        x1_new = x1+x1_otn
        y1_new = y1+y1_otn
        x2_new = x2+x2_otn
        y2_new = y2+y2_otn
        x3_new = x3+x3_otn
        y3_new = y3+y3_otn
        pos_new = [[x1_new,y1_new], [x2_new,y2_new], [x3_new,y3_new]]

        # Описание вариации данного движения
        counter += 1
        log.append([counter, 'NL of A1',x1_new,y1_new,'from SL of A1',x1_otn, y1_otn ,
                             'NL of A2',x2_new,y2_new,'from SL of A2',x2_otn, y2_otn ,
                             'NL of A3',x3_new,y3_new,'from SL of A3',x3_otn, y3_otn ])

        # Создание нового ключа в базу
        tik_add = (x1_new)*1000000000000 + (y1_new)*10000000000 + (x2_new)*10000000 + (y2_new)*100000 + (x3_new)*100 + (y3_new)

        # Проверка ключей на схожесть
        if (tik_add not in graph.keys()) & (tik_add not in tik):
            tik.append(tik_add)

    graph[x1*1000000000000 + y1*10000000000 + x2*10000000 + y2*100000 + x3*100 + y3]=log
    del tik[0]
    print(len(tik), len(graph))

# Текстовый вывод комбинаций
comb=0
for coords in graph.keys():
    comb+=1
    print('\n')
    print(coords,'   ', comb, '/', len(graph.keys()))
    print('L of A1 - x1 y1:',coords//1000000000000, coords%1000000000000//10000000000)
    print('L of A2 - x2 y2:',coords%10000000000//10000000, coords%10000000//100000)
    print('L of A3 - x3 y3:',coords%100000//100, coords%100//1)
    for count in range(len(graph[coords])):
        print(graph[coords][count])


# GUI для более удобного отображения карты
class ImageViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        self.fig = plt.figure(figsize=(4,4), dpi = 100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()

        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        filemenu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open",command=self.open_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Open Image", filetypes=[("Image Files", "*.png *.jpg *.bmp *.jpeg")])
        if file_path:
            self.ax.clear()
            img = plt.imread(file_path)
            self.ax.imshow(img)
            self.canvas.draw()

root = tk.Tk()
iv = ImageViewer(root)
root.mainloop()