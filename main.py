# Суть: рассмотреть ограниченную плоскость, на которой задано в каком-то диапазоне множество Мандельброта 
# Плоскость представлена в виде сетчатой матрицы для рассотрения каждой точки
# Проходя по каждой точке мы запоминаем её результат в матрице, смотрим, не убегает ли точка в бесконечность
# Чем больше итераций не убегает точка, тем она темнее на плоскости

import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

width = 800    # ширина в пикселях
height = 600   # высота в пикселях

# диапазон существования нашего фрактала
x_min = -0.745   
x_max = 0.11  
y_min = -0.735  
y_max = 0.12   

# максимальное число итераций
max_iter_count = 100

# Линевание плоскости
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)

# Превращение в сетку
X, Y = np.meshgrid(x, y)

# Смысл сетки: C[i,j] — комплексное число c для пикселя в строке i, столбце j
C = X + 1j * Y 

# Матрица для хранения значений точек
Z = np.zeros_like(C, dtype=np.complex128)

# Матрица для хранения итераций, когда у нас точка вылетит в бесконечность
M = np.zeros(C.shape, dtype=np.int32)    

for i in range(max_iter_count):
    # Формула Мандельброта
    Z = Z**2 + C 

    # Убежала ли наша точка?
    escaped = (np.abs(Z) > 2) & (M == 0)

    # Поскольку escaped наше полотно, показывающее, какая точка убежала, оно заставляет матрицу поменять значение
    # убежавшей точки на номер итерации, что повлияет на оттенок точки
    M[escaped] = i

# Точки, которые не убежали, получают максимальное значение
M[M == 0] = max_iter_count

# Визуализация
plt.figure(figsize=(10, 8))

cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
gradient = random.choice(cmaps)

plt.imshow(M.T, cmap='hot')

plt.show()