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

```python id="-LLLq0h_kUW2"
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
from IPython.display import Audio
```

<!-- #region id="MV-q0QtMT9Vb" -->
# Nachtrag zum Übungsblatt: STFT | Heisenberg
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sL8LAzXzT921" outputId="27ffdc19-e63c-49ca-cf02-20dbf064538d"
sr = 8000
window_size = 266 # trying out different window-sizes
nyquist = sr / 2

f_res = sr / window_size

print('Freq-resolution:\t:', f_res)

np.arange(0, nyquist, f_res) # this would be the frequencies for each bin with the given window size
```

<!-- #region id="NTqJMxfZqlqM" -->
# Logarithmus-Frequenz-Skaliertes Spektrogramm

(Log-Freq-Spec)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="71bGgab3q3eq" outputId="900ad408-6fb0-46c2-bda8-2becd6a5e9a9"
!curl https://cdn.freesound.org/previews/94/94812_29541-lq.mp3 -o piano_scale.mp3
```

```python colab={"base_uri": "https://localhost:8080/"} id="yRD0zF15raAy" outputId="3d2be97a-7299-4691-f1a9-f9484e1f7f39"
x, sr = lr.load('piano_scale.mp3', sr=None) # sr=None for file-based sr
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="6kod0xzirzmy" outputId="6bd7bfd8-6fec-48ff-8a1d-a67d937abbb0"
display(Audio(x, rate=sr))
```

```python id="RBsPj0mNr5l_"
window_size = 4096

stft = lr.stft(x, n_fft=window_size)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="_lv22D6QsDm9" outputId="27a3a50e-2e58-43cf-e37b-e5723e1e835a"
plt.figure(figsize=(20, 10))
plt.imshow(np.abs(stft), aspect='auto', origin='lower')
plt.show()
```

<!-- #region id="YkvPTNMTsSRW" -->
## Side-Note: Log-Gamma-Compression
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="nQ0lVm5HsYCa" outputId="9dbf5ec5-5942-46eb-9d7f-386388e4d6a7"
gamma = 1
stft_g = np.log10(1 + np.abs(stft) * gamma)

plt.figure(figsize=(20, 10))
plt.imshow(np.abs(stft_g), aspect='auto', origin='lower')
plt.show()
```

<!-- #region id="Kj8aNMe4treo" -->
## Fahrplan

- Welche bins entsprechen einem gegebenen Pitch `p`?
    - (p +- 0.5) range
    - midi_to_frequency
    - f_coef in stft
- sum bins per pitch
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Dr1Ehd3HtnDl" outputId="2150bae9-16cd-4f30-d5db-fd2aaf4b7cf6"
def midi_to_frequency(p):
    return 440 * 2 ** ((p - 69) / 12)

print(midi_to_frequency(69))
print(midi_to_frequency(68))
print(midi_to_frequency(69 - 12))
print(midi_to_frequency(69 - 12 - 1))
print(midi_to_frequency(60))
```

```python colab={"base_uri": "https://localhost:8080/"} id="N0LootsZvClc" outputId="7e746038-77ca-4069-aa30-903a7637cfe8"
def f_coef(k, sr, window_size):
    return sr * k / window_size

print(f_coef(k=1, sr=sr, window_size=window_size))
```

```python colab={"base_uri": "https://localhost:8080/"} id="hsy1rAGyxDdV" outputId="51becf66-4fbe-4874-de59-603bbbd1b7f4"
def freqs_in_dft(p, window_size, sr):

    f_coefs = f_coef(k=np.arange(0, window_size), sr=sr, window_size=window_size)

    freqs = np.array([])

    for f in f_coefs:
        if (f > midi_to_frequency(p - 0.5) and (f < midi_to_frequency(p + 0.5))):
            freqs = np.append(freqs, f)
    
    return freqs
          
print(freqs_in_dft(69, window_size, sr))
print(freqs_in_dft(68, window_size, sr))
print(freqs_in_dft(69 - 12, window_size, sr))
print(freqs_in_dft(69 - 12 * 2, window_size, sr))
```

```python colab={"base_uri": "https://localhost:8080/"} id="3tlMy8j80zZZ" outputId="cb920ef5-6154-4f0b-f43e-cc52e7107ee8"
def ks_in_dft(p, window_size, sr):

    f_coefs = f_coef(k=np.arange(0, window_size // 2), sr=sr, window_size=window_size)

    ks = np.array([], dtype=int)

    for k, f in enumerate(f_coefs):
        if (f > midi_to_frequency(p - 0.5) and (f < midi_to_frequency(p + 0.5))):
            ks = np.append(ks, k)
    
    return ks
          
print(ks_in_dft(69, window_size, sr))
print(ks_in_dft(68, window_size, sr))
print(ks_in_dft(69 - 12, window_size, sr))
print(ks_in_dft(69 - 12 * 2, window_size, sr))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="WJVSLSKY0MOx" outputId="ab9c3c11-770b-4cf0-e717-ef1685300db3"
def logfreq_from_stft(stft, sr, window_size):

    logfreq = np.zeros((128, stft.shape[1]))

    for p in range(128):
    
        ks = ks_in_dft(p, window_size, sr)
    
        for k in ks:
            logfreq[p] = logfreq[p] + stft[k]
    
    return logfreq

logfreq = logfreq_from_stft(np.abs(stft), sr, window_size)

plt.figure(figsize=(20, 10))
plt.imshow(logfreq, aspect='auto', origin='lower')
plt.show()
```
