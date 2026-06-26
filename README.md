# Audio-Driven Mandelbrot Set Visualization

This project explores the intersection of audio signal processing and fractal geometry. It generates a visual representation of the **Mandelbrot set**, where the parameters of the fractal (coordinate range, zoom, and color scheme) are dynamically adjusted based on the characteristics of an input audio file.

## Core Concept

The Mandelbrot set is defined by the iterative equation $Z_{n+1} = Z_n^2 + C$. 
- We represent the complex plane as a grid of points.
- For each point $C$, we iterate the formula. If the sequence diverges (escapes to infinity), the iteration count at the moment of escape determines its color.
- **Audio Integration:** Instead of a static visualization, we extract audio features (RMS for loudness and MFCCs for spectral content) to map the sound landscape to specific regions of the Mandelbrot set.

## Features

- **Audio Analysis:** Uses `librosa` to process audio files, trim silence, and extract meaningful features.
- **Dynamic Mapping:**
    - **MFCCs:** Map frequency characteristics to the fractal's spatial coordinates ($x$ and $y$ ranges).
    - **RMS (Loudness):** Maps volume levels to different `matplotlib` color maps.
- **Mathematical Rendering:** Efficiently calculates the set using `numpy` vectorization.

## Prerequisites

You will need Python 3.x installed. The project relies on the following libraries:

```bash
pip install numpy matplotlib librosa
