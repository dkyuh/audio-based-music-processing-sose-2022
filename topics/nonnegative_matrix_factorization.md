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

<!-- #region id="nON94VxHx6We" -->
# Automatic Music Transcription using Non-negative Matri Factorization (NMF)
<!-- #endregion -->

```python id="JMLWapMroqyF" executionInfo={"status": "ok", "timestamp": 1656427694160, "user_tz": -120, "elapsed": 3131, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}}
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr, librosa.display
from IPython.display import Audio
import sklearn
```

<!-- #region id="IdOS5i_zxvZW" -->
Annahme: Es existieren zwei Matrizen $W$ und $H$, sodass sie miteinander multipliziert (in etwa) $V$ ergeben.

$V \approx W \cdot H$

Dimensionen:

$V(K x N), W(K x R), H(R x N)$

wobei $R$ frei gewählt werden kann.

(Siehe auch den nächsten (kommentierten) Code-Block als Darstellung). Die Matrizen $W$ und $H$ müssen laut der Matrizenmultiplikation jeweils eine Dimension mit V gemeinsam haben. Die jeweils übrige Dimension ($R$) kann frei gewählt werden.

[D. Lee and H. S. Seung, “Algorithms for non-negative matrix factorization,” Advances in neural information processing systems, vol. 13, 2000.](https://proceedings.neurips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)
<!-- #endregion -->

```python id="KMccuOiMxzyC" executionInfo={"status": "ok", "timestamp": 1656427694161, "user_tz": -120, "elapsed": 19, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}}
# K = 5
# N = 4

#            N
#          H H H H  R
#          H H H H
#     R
#    W W   V V V V
#    W W   V V V V
# K  W W   V V V V
#    W W   V V V V
#    W W   V V V V

# bei uns: V := STFT
```

```python colab={"base_uri": "https://localhost:8080/", "height": 106} id="W-GpEQNBrJMO" executionInfo={"status": "ok", "timestamp": 1656427697272, "user_tz": -120, "elapsed": 3127, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="f713948c-adc7-4fb5-81aa-42f12db85b4f"
x, sr = lr.load('../data/piano_scale.mp3')

display(Audio(x, rate=sr))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 703} id="PvUkZ5JJr1-j" executionInfo={"status": "ok", "timestamp": 1656427698558, "user_tz": -120, "elapsed": 1310, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="9e02ee4b-f0b7-4f6b-f176-def71b72a869"
window_size = 2048

stft = lr.stft(x, n_fft=window_size, window='hann')
V = np.abs(stft)
R = 8
K = V.shape[0]
N = V.shape[1]

# random initialisierung von W & H
W_init = np.random.rand(K, R)
H_init = np.random.rand(R, N)

V_approx = np.dot(W_init, H_init)

gamma = 10
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 2)
plt.title('H_init')
lr.display.specshow(np.log10(1 + H_init * gamma))
plt.subplot(2, 2, 3)
plt.title('W_init')
lr.display.specshow(np.log10(1 + W_init * gamma))
plt.subplot(2, 2, 4)
plt.title('V_approx')
lr.display.specshow(np.log10(1 + V_approx * gamma))
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 738} id="whW3XOQXt9ZM" executionInfo={"status": "ok", "timestamp": 1656427719874, "user_tz": -120, "elapsed": 3457, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="75926569-fc86-454c-f432-c0943270a0dc"
x, sr = lr.load('../data/bach_prelude_1_goul.aiff')
display(Audio(x, rate=sr))

window_size = 2048

stft = lr.stft(x, n_fft=window_size)
V = np.abs(stft)
R = 10
# K = V.shape[0]
# N = V.shape[1]

model = sklearn.decomposition.NMF(n_components=R, init='random', solver='mu')
W = model.fit_transform(V)
H = model.components_

V_approx = np.dot(W, H)

gamma = 10
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 2)
plt.title('H')
lr.display.specshow(np.log10(1 + H * gamma))
plt.subplot(2, 2, 3)
plt.title('W')
lr.display.specshow(np.log10(1 + W[:100] * gamma))
plt.subplot(2, 2, 4)
plt.title('V_approx')
lr.display.specshow(np.log10(1 + V_approx * gamma))
plt.show()
```

<!-- #region id="OQzdwe7dyV2c" -->
## Approximierte STFT anhören
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6N3D6QV-wdzy" executionInfo={"status": "ok", "timestamp": 1656427852286, "user_tz": -120, "elapsed": 251, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="dbd1e6f9-3c12-4309-8eeb-50490b7c6063"
# beispielhafte ueberlegungen, um magnituden und phasen zu komplexen
# zahlen zusammenzufuehren

c = 2 + 2j
print(c)

mag = np.sqrt(8)
print(mag)

angle_rad = np.arctan(2 / 2)
print(angle_rad, np.pi / 4)
angle_deg = angle_rad / np.pi * 180
print(angle_deg)

print(np.exp(1j * angle_rad) * mag)

print((np.cos(angle_rad) + 1j * np.sin(angle_rad)) * mag)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="x5G_JZAiwV1X" executionInfo={"status": "ok", "timestamp": 1656427865564, "user_tz": -120, "elapsed": 2702, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="597e7517-af15-454c-d308-beae2f2dade2"
x_new = lr.istft(V_approx * np.exp(1j * np.angle(stft)))

display(Audio(x_new, rate=sr))
```

<!-- #region id="7A0mO7H1y6Ek" -->
## Pitch-informed W-Initialization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="sSUKtZBI2mqU" executionInfo={"status": "ok", "timestamp": 1656427963211, "user_tz": -120, "elapsed": 525, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="98b31019-7062-4024-fd28-70a142d34975"
def pitch_template(pitch, sr, window_size, tol_pitch=0.05):
    template = np.zeros(K)
    freq_res = sr / window_size
    pitch_freq = 440 * 2 ** ((pitch - 69) / 12)
    max_freq = sr / 2
    max_order = int(max_freq / pitch_freq)

    for order in range(1, max_order):
        min_idx = int(pitch_freq * order / freq_res * (1 - tol_pitch))
        max_idx = int(pitch_freq * order / freq_res * (1 + tol_pitch))
        template[min_idx : max_idx] = 1 / order
    
    return template

plt.figure(figsize=(12, 4))
plt.plot(pitch_template(69, sr, window_size))
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 687} id="mFTDatnjzATe" executionInfo={"status": "ok", "timestamp": 1656427980020, "user_tz": -120, "elapsed": 505, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="b14d620c-4542-45d3-e34a-89ae0ffee9a3"
pitch_set = np.array([59, 60, 62, 64, 67, 69, 72, 74, 76, 77])

W_init = np.zeros((K, R))
for r in range(R):
    pitch = pitch_set[r]
    W_init[:, r] = pitch_template(pitch, sr, window_size, tol_pitch=0.08)

plt.figure(figsize=(12, 12))
lr.display.specshow(np.log10(1 + W_init[:400] * gamma))
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 738} id="g0oSdTkf5DvT" executionInfo={"status": "ok", "timestamp": 1656428041384, "user_tz": -120, "elapsed": 3123, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="a6c89129-a49f-4ae2-f66b-0735a9529514"
x, sr = lr.load('../data/bach_prelude_1_goul.aiff')
display(Audio(x, rate=sr))

window_size = 2048

stft = lr.stft(x, n_fft=window_size)
V = np.abs(stft)
V = V.astype(np.float64)

pitch_set = np.array([59, 60, 62, 64, 67, 69, 72, 74, 76, 77])
R = pitch_set.size
K = V.shape[0]
N = V.shape[1]

W_init = np.zeros((K, R))
for r in range(R):
    pitch = pitch_set[r]
    W_init[:, r] = pitch_template(pitch, sr, window_size, tol_pitch=0.08)

H_init = np.random.rand(R, N)

model = sklearn.decomposition.NMF(n_components=R, init='custom', solver='mu')
W = model.fit_transform(V, H=H_init, W=W_init)
H = model.components_

V_approx = np.dot(W, H)

gamma = 10
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 2)
plt.title('H')
lr.display.specshow(np.log10(1 + H * gamma))
plt.subplot(2, 2, 3)
plt.title('W')
lr.display.specshow(np.log10(1 + W[:200] * gamma))
plt.subplot(2, 2, 4)
plt.title('V_approx')
lr.display.specshow(np.log10(1 + V_approx * gamma))
plt.show()
```

<!-- #region id="eJW9JXqrznYY" -->
### Sonification
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367, "output_embedded_package_id": "1oXXLtKeqy-rdg-XjG5AzkDig1YdZsxd8"} id="f6CTyp8H6rgn" executionInfo={"status": "ok", "timestamp": 1656428088952, "user_tz": -120, "elapsed": 16034, "user": {"displayName": "Daniel Kurosch H\u00f6pfner", "userId": "06635282725808825539"}} outputId="e0f05b3c-6510-458a-e99f-fecf66c080f8"
for n in range(R):
    X = np.outer(W[:, n], H[n]) * np.angle(stft)
    x_reconstructed = lr.istft(X, length=x.size)
    display(Audio(x_reconstructed, rate=sr))
```

<!-- #region id="iPrwwRwh0OPf" -->
see also [this demo](https://youtu.be/g1H-7773gpo) by Audio Labs Erlangen
<!-- #endregion -->
