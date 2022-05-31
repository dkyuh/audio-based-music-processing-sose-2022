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

# Komplexe Zahlen in Python | Exponentialfunktion

```python
import numpy as np
import matplotlib.pyplot as plt
```

## Komplexe Zahlen in Python

```python
# `1j` -> 'j' kommt aus der Schreibweise im Ingenieursbereich
print(1 + 1j)
```

```python
a = 2 + 3j
b = -1 + 1j


print(a)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
```

```python
inv_a = 1 / a

print(np.conjugate(a))
print(a * inv_a)
print(a.real)
print(a.imag)
```

```python
print(np.abs(a))
print(np.sqrt(a.real ** 2 + a.imag ** 2))
```

```python
print(np.angle(a) / np.pi * 180) # rad-deg umrechnung
print(np.rad2deg( np.angle(a) ))
print(np.rad2deg( np.arctan(a.imag / a.real) )) # atan
print(np.rad2deg( np.arctan2(a.imag, a.real) )) # atan2
```

```python
c = 0 + 2j

print(np.rad2deg( np.arctan2(c.imag, c.real) )) 
# hier lohnt sich die anwendung der hilfsfunktion atan2
```

```python
# atan wirft einen DivisionByZero-Error
print(np.rad2deg( np.arctan(c.imag / c.real) ))
```

## Exponentialfunktion

```python
print(np.exp(1))
```

### Eulersche Identität

$\mathrm{e}^{\pi\,\mathrm{i}} = - 1$

$e^x := \exp(x) := \lim_{n \to \infty} \left( 1 + \frac xn \right)^n$

```python
def own_exp(x, n=10**10): # per default sehr hohes n waehlen
    return (1 + x / n) ** n
```

```python
print('e^1 --> Eulersche Zahl (Annäherung):\n', own_exp(1), '\n')
print('e^x / exp(x):\n', own_exp(3), '\n')

print('2^x / exp( ln(2) * x ):\n', np.round(own_exp(np.log(2) * 4), 2), '\n')

print('e^pi:\n', np.round(own_exp(np.pi), 2), '\n')
print('e^i:\n', np.round(own_exp(1j), 2), '\n')
print('e^(pi * i):\n', np.round(own_exp(np.pi * 1j), 2), '\n')
```

```python
print(own_exp(1)) # eulersche zahl
```

```python
# stueckweise annaeherung, indem `n` immer hoeher gewaehlt wird
print(np.round(own_exp(np.pi * 1j, n=3), 2))
```

```python
# visualisierung fuer inkrementierende n

for n in range(1, 200):
    x = np.pi * 1j
    exp_approx = (1 + x / n) ** np.arange(n + 1)
    zeros = np.zeros_like(exp_approx, dtype=float)

    plt.figure(figsize=(12, 12))
    plt.title('n = %d' % (n))
    plt.quiver(zeros, zeros, exp_approx.real, exp_approx.imag, angles='xy', scale_units='xy', scale=1)
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.grid()
#     plt.savefig('render/exp_%03d.png' % (n))
    plt.show()
```

```python
!convert render/*.png -delay 1 -loop 0 anim.gif
```

## Eulersche Formel / Relation

$e^{t \cdot 2 \pi i} = cos(t \cdot 2 \pi) + i \cdot sin(t \cdot 2 \pi)$

```python
n = np.arange(0, 1)
e = np.exp(1j * t * 2 * np.pi)


plt.figure(figsize=(5 * 2, 5 * 2))
p1 = plt.subplot(2, 2, 1)
p2 = plt.subplot(2, 2, 2)
p3 = plt.subplot(2, 2, 3)
cm = plt.cm.get_cmap('viridis')
p1.set_title('E-Funktion')
p1.set_ylabel('Imag')
p1.set_xlabel('Real')
p2.set_title('Sin (Im)')
p3.set_title('Cos (Re)')

for color, e_, t_ in zip(cm(t), e, t):
    p1.plot(e_.real, e_.imag, 'o', color=color)
    p2.plot(t_, e_.imag, 'o', color=color)
    p3.plot(e_.real, np.abs(1 - t_), 'o', color=color)

plt.tight_layout()
plt.show()
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
