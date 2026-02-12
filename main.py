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
rms = librosa.feature.rms(y=y_trim)
rms_dB = librosa.amplitude_to_db(rms)   # Перевёл в dB
rms_mean = rms_dB.mean()

width = 800    # ширина в пикселях
height = 600   # высота в пикселях


# диапазон существования нашего фрактала
x_min = -2.5   
x_max = 1.5  
y_min = -2
y_max = 2  

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = mfcc.mean(axis=1)
mfcc_1 = mfcc_mean[1]
mfcc_2 = mfcc_mean[2]

match mfcc_1:  # яркость
    case _ if mfcc_1 < -30:
        x_min, x_max = -1.5, -0.8  # тёмные звуки — левая часть
    case _ if mfcc_1 < 0:
        x_min, x_max = -0.9, -0.3  # средне-тёмные
    case _ if mfcc_1 < 30:
        x_min, x_max = -0.4, 0.2   # средне-светлые
    case _:
        x_min, x_max = 0.1, 0.7    # светлые звуки — правая часть

match mfcc_2:  # ширина спектра
    case _ if mfcc_2 < -20:
        y_min, y_max = -0.6, 0.0   # узкие звуки — низ
    case _ if mfcc_2 < 0:
        y_min, y_max = -0.3, 0.3   # средние
    case _:
        y_min, y_max = 0.0, 0.6    # широкие звуки — верх

# максимальное число итераций
max_iter_count = 50

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
cmaps = ['coolwarm', 'viridis', 'hot', 'gray']

index = 0

match rms_mean:
    case _ if rms_mean < -30:
        index = 0
    case _ if rms_mean < -25:
        index = 1
    case _ if rms_mean < -15:
        index = 2
    case _:
        index = 3

gradient = cmaps[index]
print(index)

fig, ax = plt.subplots(1, 2, figsize=(25, 8))
ax[0].plot(rms_dB[0])
ax[0].set_title("График громкости")
ax[1].imshow(M, cmap=gradient)

plt.show()