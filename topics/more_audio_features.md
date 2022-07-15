---
jupyter:
  jupytext:
    formats: md,ipynb
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

<!-- #region id="9PfpllGMu2ej" -->
# Imports
<!-- #endregion -->

```python id="v1SIJC5hu2eo"
import numpy as np, matplotlib.pyplot as plt, librosa as lr, librosa.display
from IPython.display import Audio
```

<!-- #region id="zNwMVA7lu2ep" -->
# LogFreq
<!-- #endregion -->

```python id="l64ABqcNu2eq"
def midi_to_frequency(p):
    return 440 * 2 ** ((p - 69) / 12)

def calc_f_coef(k, sr, N):
    return k * sr / N

def ks_in_dft(p, sr, N):
    ks = []
    num_ks = N // 2
    
    for k in range(num_ks):
        
        f_coef = calc_f_coef(k, sr, N)
        
        if (midi_to_frequency(p - 0.5) < f_coef) and (f_coef < midi_to_frequency(p + 0.5)):
            ks.append(k)
            
    return ks

def stft_to_logfreq(stft, sr):
    logfreq = np.zeros((128, stft.shape[1]))
    window_size = stft.shape[0]

    for p in range(128):
        ks = ks_in_dft(p, sr, window_size)
        logfreq[p, :] = np.sum(np.abs(stft[ks, :]), axis=0)
    
    return logfreq
```

```python id="FRUdeUjf4EYv" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="ba4126f3-e798-4a66-9d55-2a893fdd18f7"
x_piano, sr_piano = lr.load('piano_scale.wav', sr=None)
x_paris, sr_paris = lr.load('paris.wav', sr=None)

length_piano = x_piano.size / sr_piano
length_paris = x_paris.size / sr_paris

display(Audio(x_piano, rate=sr_piano, autoplay=True))
display(Audio(x_paris, rate=sr_paris, autoplay=True))

stft_piano = lr.stft(x_piano)
logfreq_piano = stft_to_logfreq(stft_piano, sr_piano)
stft_paris = lr.stft(x_paris)
logfreq_paris = stft_to_logfreq(stft_paris, sr_paris)


gamma = 1
plt.figure(figsize=(24, 18))
plt.imshow(np.log10(1 + np.abs(logfreq_piano) * gamma), aspect='auto', origin='lower', extent=[0, length_piano, 0, 128])
plt.show()
plt.figure(figsize=(24, 18))
plt.imshow(np.log10(1 + np.abs(logfreq_paris) * gamma), aspect='auto', origin='lower', extent=[0, length_paris, 0, 128])
plt.show()
```

<!-- #region id="wTIO2GA_u2er" -->
# Chroma
<!-- #endregion -->

```python id="s4EvR6Spu2er" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="fefbedf8-2c04-4039-84b1-f335f919593e"
def logfreq_to_chroma(logfreq):
    chroma = np.zeros((12, logfreq.shape[1]))
    for p in range(128):
        chroma[p % 12, :] = chroma[p % 12, :] + logfreq[p, :]
    return chroma

chroma_piano = logfreq_to_chroma(logfreq_piano)
chroma_paris = logfreq_to_chroma(logfreq_paris)

gamma = 1
plt.figure(figsize=(24, 18))
plt.imshow(np.log10(1 + np.abs(chroma_piano) * gamma), aspect='auto', origin='lower', extent=[0, length_piano, 0, 128])
plt.show()
plt.figure(figsize=(24, 18))
 # macht wenig sinn, aber der vollstaendigkeit halber:
plt.imshow(np.log10(1 + np.abs(chroma_paris) * gamma), aspect='auto', origin='lower', extent=[0, length_paris, 0, 128])
plt.show()
```

<!-- #region id="EJmfJLt8u2er" -->
# Helpers
<!-- #endregion -->

```python id="EMRn2ONCu2es"
def plot_feature_waveform(feature, x, normalize=True):
    plt.subplot(2, 1, 1)
    plt.plot(x)
    t = np.linspace(0, x.size, feature.size)
    if normalize:
        plt.plot(t, feature / np.max(np.abs(feature)))
    else:
        plt.plot(t, feature)
    plt.ylim(-1.1, 1.1)
    plt.subplot(2, 1, 2)
    plt.plot(feature)
    plt.tight_layout()
    plt.show()
    
def plot_feature_stft(feature, stft, sr, gamma=1):
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    lr.display.specshow(np.log(1 + stft * gamma), y_coords=lr.fft_frequencies(sr, 2048))
    plt.plot(feature)
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="gQx1oUils9wV" outputId="634657fd-7c0d-4936-808f-8b9e3f2f5413"
a = np.arange(30)

print(a, '\n')

a_framed = lr.util.frame(a, 6, 2)

print(a_framed, '\n')

np.sum(a_framed, axis=0)

# 15, 27, 39, ...
```

<!-- #region id="5cC8ux0Ru2et" -->
# Root Mean Square Energy (Wiederholung)

${\displaystyle rms = \sqrt{\dfrac{1}{n} \sum^{n}_{i=1}x_{i}^{2}} = \sqrt{\dfrac{x_{1}^{2} + x_{2}^{2} + ... + x_{n}^{2}}{n}}}$
<!-- #endregion -->

```python id="2rPh8DAcu2et" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="6aafdb66-05f4-4885-98c4-aed487aa59fe"
def calc_rms_energy_double_loop(x, hop_size=512, window_size=2048):
    num_windows = (x.size - window_size) // hop_size
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        for win_samp_idx in range(window_size):
            rms[i] += x[i * hop_size + win_samp_idx] ** 2 / window_size
    rms = np.sqrt(rms)
    return rms

rms_piano = calc_rms_energy_double_loop(x_piano)
plot_feature_waveform(rms_piano, x_piano, normalize=False)
rms_paris = calc_rms_energy_double_loop(x_paris)
plot_feature_waveform(rms_paris, x_paris, normalize=False)

def calc_rms_energy_loop(x, hop_size=512, window_size=2048):
    num_windows = (x.size - window_size) // hop_size
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        win = x[i * hop_size : i * hop_size + window_size]
        rms[i] = np.sqrt(np.mean(win ** 2))
    return rms

rms_piano = calc_rms_energy_loop(x_piano)
plot_feature_waveform(rms_piano, x_piano, normalize=False)
rms_paris = calc_rms_energy_loop(x_paris)
plot_feature_waveform(rms_paris, x_paris, normalize=False)


def calc_rms_energy_framed(x, hop_size=512, window_size=2048):
    x_framed = lr.util.frame(x, window_size, hop_size)
    rms = np.sqrt(np.mean(x_framed ** 2, axis=0))
    return rms

rms_piano = calc_rms_energy_framed(x_piano)
plot_feature_waveform(rms_piano, x_piano, normalize=False)
rms_paris = calc_rms_energy_framed(x_paris)
plot_feature_waveform(rms_paris, x_paris, normalize=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7fva-ZdpxghB" outputId="e8fab13c-b355-4c51-b46c-25bdf690301a"
%timeit calc_rms_energy_double_loop(x_piano)
%timeit calc_rms_energy_loop(x_piano)
%timeit calc_rms_energy_framed(x_piano)
```

<!-- #region id="fINDnsXlu2es" -->
# Zero-Crossing-Rate

${\displaystyle zcr = {\frac{1}{N}\sum^{N}_{i = 1} | \mathrm{sign}[x(n + i)] - \mathrm{sign}[x(n + i - 1)]|}}$

${\displaystyle zcr = {\frac{1}{N}\sum^{N}_{i = 1} | \mathrm{sign}[x(n + i + 1)] - \mathrm{sign}[x(n + i)]|}}$

$
\mathrm{sign}(x) = \left\{
\begin{array}{ll}
    +1 & \mathrm{if} \ x > 0\\
    0  & \mathrm{if} \ x = 0\\
    -1 & \mathrm{if} \ x < 0
\end{array}
\right.
$
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dg7UkmIFyhtW" outputId="d216b635-2ba1-4d1f-f49d-9f2b99f84025"
print(np.sign(-100239))
print(np.sign(100239))
print(np.sign(0))


np.arange(10)
```

```python id="mzDYGceEu2et" colab={"base_uri": "https://localhost:8080/", "height": 862} outputId="16b5e587-270c-4e2e-9244-0c53eddbcaad"
def calc_zero_crossing_rate(x, hop_size=512, window_size=2048):
    num_windows = (x.size - window_size) // hop_size
    zcr = np.zeros(num_windows)
    for i in range(num_windows):
        win = x[i * hop_size : i * hop_size + window_size]
        zcr[i] = np.mean(np.abs(np.sign(win[1:]) - np.sign(win[:-1])))
    return zcr

zcr_piano = calc_zero_crossing_rate(x_piano + 0.01)
plot_feature_waveform(zcr_piano, x_piano)
plot_feature_stft(zcr_piano * sr_piano / 2, np.abs(stft_piano), sr_piano)
zcr_paris = calc_zero_crossing_rate(x_paris + 0.01)
plot_feature_waveform(zcr_paris, x_paris)
```

<!-- #region id="nZ4Q5GDDu2et" -->
# Spectral Centroid

Schwerpunkt der Magnituden-Frequenzen im FFT-Fenster.

${\displaystyle \mathrm{centroid} = \dfrac{\sum^{N-1}_{k=0}\mathrm{freqs}(n)X(k)}{\sum^{N-1}_{k=0}X(k)}}$
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3u6c4zRZ2mG0" outputId="a669c47f-cf6e-42cd-aa4e-de85d5f0ab65"
f = np.arange(1, 7)

print(f)

print(a_framed)

print(f[:, np.newaxis] * a_framed)
```

```python id="FYuS5HR6u2eu" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="2f46ffcb-c460-4615-ec5d-905f202b2004"
def calc_spectral_centroid(stft, sr):
    freqs = lr.fft_frequencies(sr, stft.shape[0] * 2 - 1)
    spec_cent = np.sum(freqs[:, np.newaxis] * stft, axis=0) / np.sum(stft, axis=0)
    return spec_cent

spec_cent_piano = calc_spectral_centroid(np.abs(lr.stft(x_piano + 0.05)), sr_piano)
plot_feature_waveform(spec_cent_piano, x_piano)
plot_feature_stft(spec_cent_piano, np.abs(stft_piano), sr_piano)

spec_cent_paris = calc_spectral_centroid(np.abs(lr.stft(x_paris + 0.05)), sr_paris)
plot_feature_waveform(spec_cent_paris, x_paris)
plot_feature_stft(spec_cent_paris, np.abs(stft_paris), sr_paris)
```

<!-- #region id="uHFOlSdIu2eu" -->
# Spectral Bandwidth / Spread

Deutet die "Breite", bzw. Verteilung der energiereichen/energie-gleichen Frequenzkoeffizienten des Spektrums zu einem Zeitpunkt im Fenster an.

${\displaystyle \mathrm{bandwidth} = \sqrt{\dfrac{\sum_{n=0}^{N-1}(\mathrm{freqs}(n) - \mathrm{centroid})^2X(n)}{\sum_{n=0}^{N-1}X(n)}}}$
<!-- #endregion -->

```python id="p-2mz389u2eu" colab={"base_uri": "https://localhost:8080/", "height": 598} outputId="9eb0fa1d-1a3f-4ade-e5d8-da05df9b864d"
def calc_spectral_bandwidth(stft, sr):
    freqs = lr.fft_frequencies(sr, stft.shape[0] * 2 - 1)
    spec_cent = calc_spectral_centroid(stft, sr)
    spec_band = np.sqrt(np.sum( ((freqs[:, np.newaxis] - spec_cent) ** 2) * stft , axis=0) / np.sum(stft, axis=0))
    return spec_band

spec_band_piano = calc_spectral_bandwidth(np.abs(lr.stft(x_piano + 0.05)), sr_piano)
plot_feature_waveform(spec_band_piano, x_piano)
spec_band_paris = calc_spectral_bandwidth(np.abs(lr.stft(x_paris + 0.05)), sr_paris)
plot_feature_waveform(spec_band_paris, x_paris)
```

<!-- #region id="UqlDiX30u2eu" -->
## Combined Visualisation
<!-- #endregion -->

```python id="9iMB8TOwu2eu" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="8efd42b9-49eb-4c2b-fde6-91306b550191"
# t_piano = np.linspace(0, x_piano.size, spec_band_piano.size)
plt.figure(figsize=(24, 12))
lr.display.specshow(np.log(1 + np.abs(stft_piano)), y_coords=lr.fft_frequencies(sr_piano, 2048))
spec_cent_piano = calc_spectral_centroid(np.abs(lr.stft(x_piano + 0.05)), sr_piano)
plt.plot(spec_cent_piano)
plt.plot(spec_cent_piano + spec_band_piano)
plt.plot(spec_cent_piano - spec_band_piano)
plt.show()

# t_piano = np.linspace(0, x_piano.size, spec_band_piano.size)
plt.figure(figsize=(24, 12))
lr.display.specshow(np.log(1 + np.abs(stft_paris)), y_coords=lr.fft_frequencies(sr_paris, 2048))
spec_cent_paris = calc_spectral_centroid(np.abs(lr.stft(x_paris + 0.05)), sr_paris)
plt.plot(spec_cent_paris)
plt.plot(spec_cent_paris + spec_band_paris)
plt.plot(spec_cent_paris - spec_band_paris)
plt.show()
```

# Spectral Rolloff

Zeigt den Frequenzkoeffizienten an, unter dem sich der größte Anteil Energie (in diesem Fall 90%) im Spektrum befindet.

${C \sum_{k=0}^{N-1} X_n(k)}$
<!-- #endregion -->

```python id="8rxoVQwsu2ev" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="19eaf4fc-8cf1-446e-fc62-9c9346ff5696"
def calc_spectral_rolloff(stft, sr, c=0.9):
    
    total_energy = np.sum(stft, axis=0)
    freqs = lr.fft_frequencies(sr, (stft.shape[0] - 1) * 2)
    spec_roll = np.zeros_like(total_energy)

    for n in range(stft.shape[1]):

        current_energy = 0
        k = 0 # frequenz-indizes

        while (current_energy <= c * total_energy[n]) and (k < stft.shape[0]):

            current_energy += stft[k, n]
            k += 1
        
        spec_roll[n] = freqs[k]

    return spec_roll

spec_roll_piano = calc_spectral_rolloff(np.abs(lr.stft(x_piano + 0.05)), sr_piano)
plot_feature_waveform(spec_roll_piano, x_piano)
plot_feature_stft(spec_roll_piano, np.abs(stft_piano), sr_piano)
spec_roll_paris = calc_spectral_rolloff(np.abs(lr.stft(x_paris + 0.05, n_fft=1024)), sr_paris)
plot_feature_waveform(spec_roll_paris, x_paris)
plot_feature_stft(spec_roll_paris, np.abs(lr.stft(x_paris + 0.05, n_fft=1024)), sr_paris, gamma=1)
```

<!-- #region id="AmcuQ9Qqu2ev" -->
# Spectral Flatness

Wie "flach" ist das Spektrum?

--> geräuschartige Klänge: hohe Flachheit

--> (quasi-)periodische Klänge: niedrige Flachheit

${\displaystyle \mathrm{flatness} = \dfrac{\exp\left(\frac{1}{N}\sum_{k=0}^{N-1}\log(X_n(k)^2)\right)}{\frac{1}{N}\sum_{k=0}^{N-1}X_n(k)^2}}$
<!-- #endregion -->

```python id="aVVOKnCBu2ev" colab={"base_uri": "https://localhost:8080/", "height": 577} outputId="6ec86133-58a0-4b4d-c87f-68432c8bfb0b"
def calc_spectral_flatness(stft):
    pow_stft = stft ** 2
    spec_flat = np.exp(np.mean(np.log(pow_stft), axis=0)) / np.mean(pow_stft, axis=0)
    return spec_flat

spec_flat_piano = calc_spectral_flatness(np.abs(lr.stft(x_piano + 0.05)))
plot_feature_waveform(spec_flat_piano, x_piano)
spec_flat_paris = calc_spectral_flatness(np.abs(lr.stft(x_paris + 0.05)))
plot_feature_waveform(spec_flat_paris, x_paris)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="Rfpf8OZn3yQf" outputId="4e32a99c-d21f-4256-d932-ce1427752146"
plt.figure(figsize=(20, 10))
lr.display.specshow(np.log10(1 + np.abs(stft_paris)))
plt.show()
```
