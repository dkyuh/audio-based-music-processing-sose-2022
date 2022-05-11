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

# Matplotlib

```python
# install matplotlib (if not installed)

# import sys

# !{sys.executable} -m pip install matplotlib
```

## plot-function

```python
import matplotlib.pyplot as plt
```

```python
print(a)

plt.plot(a)
```

```python
plt.plot(a, '.')
```

```python
plt.plot(a, '.-')
```

```python
plt.plot(a, 'ro')
```

```python
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
```

```python
x = np.linspace(-20, 20, num=1000)

y_1 = x
y_2 = x ** 2
y_3 = x ** 3
y_4 = 1 / x ** 2

plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)
plt.plot(x, y_4)
```

## xlim | ylim

```python
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)
plt.plot(x, y_4)

plt.ylim(-200, 200)
plt.xlim(-20, 20)
```
