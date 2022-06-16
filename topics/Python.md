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

# Python specific stuff

## Strings

```python
'\n' # new line
'\t' # tab

print('hallo', '\t', 'welt', '\n', 'hallo', '\t', 'du')
```

## Teilung ohne Rest

also see: [Windowing Time-Framing](/topics/time_framing.md#Windowing%20Time-Framing)

```python
print(13 / 5)
print(13 // 5) # --> teilung ohne rest
```

## Komplexe Zahlen in Python

- siehe [Komplexe Zahlen in Python](/topics/Mathematik-Grundlagen.md#Komplexe%20Zahlen%20in%20Python)

## enumerate

```python id="1dXbBQphrC-z" outputId="0302a517-7ee3-4109-97e0-e41925bc4941" colab={"base_uri": "https://localhost:8080/"}
r = np.random.rand(10)

i = 0

for num in r:
    print(i, num)
    i += 1

print()

for i, num in enumerate(r):
	print(i, num)
```