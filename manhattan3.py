data = [[2, 10, 7, 8, 1],
        [5, 6, 7, 10, 0],
        [10, 8, 6, 5, 1],
        [7, 2, 8, 6, 0],
        [9, 7, 9, 8, 1],
        [5, 3, 7, 4, 0],
        [8, 5, 2, 7, 1]]

test = [7, 8, 5, 9]

def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

jarak = []

for row in data:
    d = manhattan(test, row[:4])
    jarak.append([d, row[4]])

jarak.sort()
tetangga = jarak[:3]
label0 = 0
label1 = 0

for d, label in tetangga:
    if label == 0:
        label0 += 1
    else:
        label1 += 1

if label0 > label1:
    print("Label Kelas 0")
else:
    print("Label Kelas 1")