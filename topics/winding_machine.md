---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
```

## Winding Machine


also see: [enumerate](/topics/Python.md#enumerate)

```python
sr = 100
length = 1
freq1, freq2 = 2, 5
t = np.linspace(0, length, sr * length)
x = (np.sin(t * 2 * np.pi * freq1) * 0.5) + (np.sin(t * 2 * np.pi * freq2) * 0.5)
# x = (x * 0.5 + 0.5) * 0.75 + 0.25

winding_freqs = np.linspace(0, 7, 100)
mag = np.zeros_like(winding_freqs)

for i, winding_freq in enumerate(winding_freqs):
    exp_func = np.exp(t * 2 * np.pi * -1j * winding_freq)
    winded = exp_func * x
    pom = np.mean(winded) # "point of mass"
    mag[i] = np.abs(pom)
#     print(pom)


    plt.figure(figsize=(5 * 3, 5))

    plt.subplot(1, 3, 1)
    # plt.plot(exp_func.real, exp_func.imag, 'o')
    plt.plot(winded.real, winded.imag, 'o')
    plt.arrow(0, 0, pom.real, pom.imag, color='red', linewidth=3)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.subplot(1, 3, 2)
    plt.plot(x)
    plt.plot(np.cos(t * 2 * np.pi * winding_freq + np.angle(pom)))
    plt.ylim(-1.1, 1.1)

    plt.subplot(1, 3, 3)
    plt.plot(winding_freqs, mag)

    plt.show()
```

## Diskrete Fourier Transformation

([siehe engl. Wiki](https://en.wikipedia.org/wiki/Discrete_Fourier_transform))

Die diskrete Fourier Transformation transformiert eine Sequenz mit $N$ komplexen Zahlen $\left\{ \mathbf{x}_n \right\} := x_0, x_1, \ldots , x_{N - 1}$ in eine andere Sequenz aus komplexen Zahlen, $\left\{ \mathbf{X}_k \right\} := X_0, X_1, \ldots , X_{N - 1}$, wie folgt:

${ \begin{aligned} X_{k} &= \sum_{n = 0}^{N - 1} x_n \cdot e^{-i 2 \pi \frac{n}{N} k} \\ &= \sum_{n = 0}^{N - 1} x_n \cdot \left[ \cos \left( 2 \pi \frac{n}{N} k \right) - i \cdot \sin \left( 2 \pi \frac{n}{N} k \right) \right], \end{aligned} }$

## Begriffsunterscheidungen

- FT --> Fourier Transform
- DFT --> Discrete Fourier Transform

- STFT ((Discrete) Short Time Fourier Transform (--> Heisenbergsche Unschärferelation)

- FFT --> Fast Fourier Transform (schnelle Variante der DFT)
