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

# Time-Framing / Windowng
related:

STFT --> Short Time Fourier Transform benutzt Windowing/Time-Framing

Im Gegensatz dazu:

FT --> Fourier Transform

```python id="tN0TSn0kTy7t"
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
```

## ADSR-enveloped Signal erstellen

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="TUzesen1T9zR" outputId="50962dd0-1af6-49bc-eac4-505df43bb6e9"
sr = 44100
length = 1
freq = 800
t = np.linspace(0, length, int(length * sr))
x = np.sin(t * 2 * np.pi * freq)

env_part_length = int(sr * length / 4)
decay_val = 0.9
a = np.linspace(0, 1, env_part_length)
d = np.linspace(1, decay_val, env_part_length)
s = np.ones(env_part_length) * decay_val
r = np.linspace(decay_val, 0, env_part_length)
env = np.concatenate((a, d, s, r))

x = x * env

plt.plot(x)
plt.show()

display(Audio(x, rate=sr))
```

## Windowing / Time-Framing
also see: [min max](/topics/numpy.md#min%20max)

```python colab={"base_uri": "https://localhost:8080/"} id="jzllMgkdayrf" outputId="5e792d6c-cffb-4015-eba8-4a7079524588"
a = np.random.randint(0, 100, 10)

print(a)

print(np.max(a))
```

also see: [Teilung ohne Rest](/topics/Python.md#Teilung%20ohne%20Rest)

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="_6EOv3-jX94z" outputId="d7c27fe2-1d82-4363-9d2d-57c4102b4170"
# szenario: no overlap
window_size = 300

num_windows = x.size // window_size

print(num_windows)

amp_env = np.zeros(num_windows)

for i in range(num_windows):
    start_idx = i * window_size
    stop_idx = (i + 1) * window_size
    # print(i, start_idx, stop_idx)
    win = x[start_idx:stop_idx]

    amp_env[i] = np.max(win)

plt.plot(amp_env)
plt.show()
```

```python id="0BqnisQ3SjA_"
# szenario :overlap
window_size = 300
hop_size = 100

num_windows = (x.size - window_size) // hop_size

print(num_windows)

amp_env = np.zeros(num_windows)

for i in range(num_windows):
    start_idx = i * hop_size
    stop_idx = start_idx + window_size
#     print(i, start_idx, stop_idx)
    win = x[start_idx:stop_idx]

    amp_env[i] = np.max(win)

plt.plot(amp_env)
plt.show()
```

### Try out on 'real-world' file

```python id="6GNCOKEVd6F1"
import librosa as lr
```

```python colab={"base_uri": "https://localhost:8080/", "height": 106} id="r6hjh2H_fV1j" outputId="9cd0bd13-c211-43dd-e198-7e42ab1c1335"
x, sr = lr.load('../data/flute.mp3')

display(Audio(x, rate=sr))
```

```python id="a0O2c_iLgRCn"
def calc_amp_env(x, window_size, hop_size):
    num_windows = (x.size - window_size) // hop_size # teilung ohne rest --> python-topic
    # print(num_windows)
    max_env = np.zeros(num_windows)
    min_env = np.zeros(num_windows)
    mean_env = np.zeros(num_windows)

    for i in range(num_windows):
        start_idx = i * hop_size
        stop_idx = start_idx + window_size
        # print(i, start_idx, stop_idx)
        win = x[start_idx:stop_idx]

        max_env[i] = np.max(win)
        min_env[i] = np.min(win)
        mean_env[i] = np.mean(np.abs(win)) # --> numpy-topic

    return [max_env, min_env, mean_env]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="i2StTJDGglbF" outputId="2ffd9b81-1d58-42c6-8d9e-71dbdf4a47e0"
window_size = 300
hop_size = 100

max_env, min_env, mean_env = calc_amp_env(x, window_size, hop_size)

plt.plot(x)
plt.plot(np.linspace(0, x.size, max_env.size), max_env)
plt.plot(np.linspace(0, x.size, min_env.size), min_env)
plt.plot(np.linspace(0, x.size, mean_env.size), mean_env)
plt.show()
```

```python id="hbOQcSnNkCBL"
# mean
# max --> positive peak
# min --> negative peak
# rms --> root mean square

# lufs 
```

## Resources
1. 
