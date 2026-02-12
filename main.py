# Суть: рассмотреть ограниченную плоскость, на которой задано в каком-то диапазоне множество Мандельброта 
# Плоскость представлена в виде сетчатой матрицы для рассотрения каждой точки
# Проходя по каждой точке мы запоминаем её результат в матрице, смотрим, не убегает ли точка в бесконечность
# Чем больше итераций не убегает точка, тем она темнее на плоскости

import numpy as np
import matplotlib.pyplot as plt
import librosa
import random
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# Импортируем файл со звуком и дискретизируем его
audio_path = 'Sound_19154300 1633539888.mp3'
y, sr = librosa.load(audio_path)

# Обрезаем тишину (ниже -30 dB)
y_trim, _ = librosa.effects.trim(y=y, top_db=30)

# Извлекаем громкость
rms = librosa.feature.rms(y=y)
rms_dB = librosa.amplitude_to_db(rms)   # Перевёл в dB
rms_mean = rms_dB.mean()

width = 800    # ширина в пикселях
height = 600   # высота в пикселях

# диапазон существования нашего фрактала
x_min = -2.5   
x_max = 1.5  
y_min = -2
y_max = 2  

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
    Z = Z**2 + C*1.4

    # Убежала ли наша точка?
    escaped = (np.abs(Z) > 2) & (M == 0)

    # Поскольку escaped наше полотно, показывающее, какая точка убежала, оно заставляет матрицу поменять значение
    # убежавшей точки на номер итерации, что повлияет на оттенок точки
    M[escaped] = i

# Точки, которые не убежали, получают максимальное значение
M[M == 0] = max_iter_count

# Визуализация
plt.figure(figsize=(10, 8))

cmaps = ['plasma', 'inferno', 'magma', 'hot']

index = 0

match rms_mean:
    case _ if rms_mean < -30:
        index = 0
    case _ if rms_mean < -20:
        index = 1
    case _ if rms_mean < -10:
        index = 2
    case _:
        index = 3

gradient = cmaps[index]

plt.imshow(M, cmap=gradient)

plt.show()