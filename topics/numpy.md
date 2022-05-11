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

# Numpy

```python
import numpy as np
```

## Numpy Arrays vs. Listen / Tuples

```python
# numpy-arrays

# eigentlich gibt es in Python verschiedene Listen-Typen
[1, 2, 3] # Liste
```

```python
(1, 2, 3) # Tuple
```

```python
[[1, 2, 3], [4, 5, 6]] # ein Beispiel fuer eine verschachtelte Liste
```

```python
np.array([1, 2, 3]) # numpy-array (eigentlich wird hier aus einer Liste ein np-array erstellt)
```

```python
a_l = [1, 2, 3]
b_l = [4, 5, 6]

a_l + b_l # + Operation auf Listen
# concatenate
```

```python
a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])

a_np + b_np # + Operation auf np-arrays
# Operationen werden elementweise ausgefuehrt
```

## datentypen in np-arrays vs in Listen

```python
# listen sind sehr flexibel:
[1, "hallo", 1.4, [4, 5, 7]]
```

```python
np.array([1, "hallo", 1.4]) # hier wurden alle datentypen der elemente in strings ge-castet
```

```python
np.array([1, 1.4]) # hier wurden alle datentypen der elemente in floats ge-castet
```

```python
np.array([1, 1.4], dtype=str) # type-casting selbst bestimmen mit dem argument 'dtype'
```

```python
np.array([1, "hallo", 1.4, [4, 5, 7]])
```

## Recheneffizienz mit Numpy-Arrays vs. Listen

```python
# beispiel: elementweise addieren

# listen: 

added_list = []

for index in range(len(a_l)):
#     print(index)
    added_list.append(a_l[index] + b_l[index])

added_list
```

```python
def add_lists(a_l, b_l):
    
    added_list = []
    
    for index in range(len(a_l)):
        added_list.append(a_l[index] + b_l[index])
    
    return added_list
```

```python
# elementweise addieren mit sehr grossen listen

a_l = []
b_l = []

for i in range(10000000):
    a_l.append(i)
    b_l.append(i * 2)
```

```python
%timeit add_lists(a_l, b_l)
```

```python
a_np = np.array(a_l)
b_np = np.array(b_l)
```

```python
%timeit a_np + b_np
```

```python
# --> elementweise verrechnen mit np-arrays ist viel schneller als mit Listen
```

```python
# in diese np-arrays werden wir vor allem audio-daten schreiben, verrechnen, etc.
```

## Array Operationen

```python
a = np.array([1, 3, 49, 81, 5, 67])
```

```python
print(a + 1)
print(a * 2)
print(a / 2)
print(a < 6)

print(a + np.array([100, 200, 300, 400, 500, 600]))
# dimensionen und shape muessen gleich sein
```

```python
print(np.shape(a))
print(a.shape)
```

## Multidimensionale Arrays

```python
b = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
print(b)
# wird wie eine tabelle oder (in diesem fall 2d-)matrix behandelt
```

```python
print(b.shape)
```

```python
# 3d
# (4, 7, 8)

print(np.arange(10))
print(np.arange(2, 10))
print(np.arange(2, 10, 3))
      
      
np.reshape(np.arange(224), newshape=(4, 7, 8))
```

## Auf Elemente zugreifen

### einzelne Elemente

```python
print(a)

print(a[1])

print(a[-2]) # vor-vor-letztes element

print()

print(b, '\n')

print(b[1], '\n')

print(b[0][1], '\n')

print(b[0, 1], '\n')
```

```python
b_list = [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]

print(b_list)
print(b_list[0][1])
```

```python
b_list[0, 1] #funktioniert nicht bei listen
```

### mehrere Elemente

```python
print(a)
print(a[1:4]) # die rechte grenze ist exklusiv, d.h. wir haben eine element-range von 1-3 
print(a[1:])
print(a[:4])
print(a[:])
print(a[:-1]) # mit -1 die elemente von hinten zaehlen
print(a[:-2])

print()

# boolean index
print(np.array([True, False, False, True, False, True]))
print(a[ np.array([True, False, False, True, False, True]) ])
# auch hier muessen die dimensionen einander entsprechen
```

```python
print(b, '\n')

print(b[1:3], '\n')

print(b[1:], '\n')
print(b[:2], '\n')
print(b[:-1], '\n')

print(b[0:2 , 1], '\n')

print(b[0:2 , 1:2], '\n')

print(b[0:2 , 1:3], '\n')
```

### bool-index

```python
# wie kann man auf 2 und 6 zugreifen

# sieve <-> Sieb

bool_idx = np.array([ [False, True,  False],
                      [False, False, True ],
                      [False, False, False]])

print(bool_idx)

print(b[bool_idx])



print()
# bool-index eleganter erstellen

# bool_idx = (b == 2) or (b == 6) # funktioniert so nicht
                                  # (aber nur wegen syntax - der gedanke an sich fuehrt zum Ziel)
    
bool_idx = np.logical_or(   b == 2,    b == 6   )

print(bool_idx)



print()
# for-schleife

j = 1
for i in range(2):
    if (j > 2):
        break
    print(b[i, j])
    j += 1
    
    

print()
# for-schleife

for i in range(2):
    print(b[i, i + 1])

# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
```

## Arrays erstellen

```python
print(np.arange(50, 100, 4))

np.linspace # --> Shift-Tab fuer Documentation

print(np.linspace(50, 100, num=10)) # bei linspace ist das 'stop'-argument **nicht** exklusiv

print(np.linspace(50, 98, num=13))

print(np.linspace(50, 98, num=13, dtype=int))

print(np.zeros(10))

print(np.ones(10))

print(np.ones(10) * 400)
```

## datatypes

```python
print(np.linspace(50, 98, num=13, dtype=int))
# datatypes koennen in fast jeder np-funktion mit dem 'dtype'-argument bestimmt werden
```

## Pi

```python
np.pi
```

## Sine-Function

```python
print(np.sin(0))
print(np.round(np.sin(1 / 4 * np.pi), 2))
print(np.round(np.sin(1 / 2 * np.pi), 2))
print(np.round(np.sin(3 / 4 * np.pi), 2))
print(np.round(np.sin(1 * np.pi), 2))
print(np.round(np.sin(5 / 4 * np.pi), 2))
print(np.round(np.sin(3 / 2 * np.pi), 2))
print(np.round(np.sin(7 / 4 * np.pi), 2))
print(np.round(np.sin(2 * np.pi), 2))
```

## Runden

```python
print(np.round(2.4941813948519385293874934857, 0))
print(np.round(2.4941813948519385293874934857, 1))
print(np.round(2.4941813948519385293874934857, 3))
```

## min / max

also see: [Windowing Time-Framing](topics/time_framing.md#Windowing%20Time-Framing)

```python
a = np.random.randint(0, 100, 10)

print(a)

print(np.max(a))
print(np.min(a))
```
