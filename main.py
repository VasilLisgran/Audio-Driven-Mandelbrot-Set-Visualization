# Concept: examine a bounded region of the plane where the Mandelbrot set is defined
# The region is represented as a grid matrix to allow for the analysis of individual points
# We iterate through each point, store the result in the matrix, and check whether the point diverges to infinity
# The more iterations a point undergoes without diverging, the darker it appears on the plane

import numpy as np
import matplotlib.pyplot as plt
import librosa
import random
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# Import the audio file and sample it
audio_path = 'Sound_19154300 1633539888.mp3'
y, sr = librosa.load(audio_path)

# Trim silence (below -30 dB)
y_trim, _ = librosa.effects.trim(y=y, top_db=30)

# Extracting volume
rms = librosa.feature.rms(y=y_trim)
rms_dB = librosa.amplitude_to_db(rms)   # in dB
rms_mean = rms_dB.mean()

width = 800    # width in pixels
height = 600   # height in pixels


# range of existence of our fractal
x_min = -2.5   
x_max = 1.5  
y_min = -2
y_max = 2  

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = mfcc.mean(axis=1)
mfcc_1 = mfcc_mean[1] # brightness
mfcc_2 = mfcc_mean[2] # spectral width

match mfcc_1:  # brightness
    case _ if mfcc_1 < -30:
        x_min, x_max = -1.5, -0.8  # dark sounds — left side
    case _ if mfcc_1 < 0:
        x_min, x_max = -0.9, -0.3  # medium-dark
    case _ if mfcc_1 < 30:
        x_min, x_max = -0.4, 0.2   # medium-light
    case _:
        x_min, x_max = 0.1, 0.7    # bright sounds — right side

match mfcc_2:  # spectral width
    case _ if mfcc_2 < -20:
        y_min, y_max = -0.6, 0.0   # narrow sounds — low end
    case _ if mfcc_2 < 0:
        y_min, y_max = -0.3, 0.3   # middle
    case _:
        y_min, y_max = 0.0, 0.6    # wide sounds — upper register

# maximum number of iterations
max_iter_count = 50

# Lineation of the plane
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)

# Conversion to a mesh
X, Y = np.meshgrid(x, y)

# Meaning of the grid: C[i,j] — a complex number c for the pixel at row i, column j
C = X + 1j * Y 

# Matrix for storing point values
Z = np.zeros_like(C, dtype=np.complex128)

# Matrix for storing the iteration counts at which the point escapes to infinity
M = np.zeros(C.shape, dtype=np.int32)    

for i in range(max_iter_count):
    # Mandelbrot formula
    Z = Z**2 + C

    # Is point escaped?
    escaped = (np.abs(Z) > 2) & (M == 0)

    # Since `escaped` is the canvas showing which point has escaped, it causes the matrix to update the value
    # of the escaped point to the iteration number, which affects the point's color shade.
    M[escaped] = i

# Points that did not escape receive the maximum value
M[M == 0] = max_iter_count

# Visualisation
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
ax[0].set_title("Volume")
ax[1].imshow(M, cmap=gradient)

plt.show()
