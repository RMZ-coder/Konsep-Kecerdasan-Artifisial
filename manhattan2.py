import pandas as pd
import random
from random import shuffle

K = 5
4 fitur 7 kolom 9 class
random.seed(42)
print("\nMenghitung Data Kepuasan Pelanggan dengan K-Nearest Neighbors (Manhattan)" "\n")
print("-"*70, "\n")

df= pd.read_csv('datasetkepuasan/train.csv',sep=';')
print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(df.isnull().sum())
print("\nMenghapus baris yang kosong atau tidak bernilai")
df = df.dropna()
print(f"Jumlah dataset setelah data kosong terhapus: {df.shape[0]} baris, {df.shape[1]} kolom")

print("\nMenggunakan 5 fitur karena 21 fitur terlalu berat")
print("Menghapus fitur selain : Flight Distance, Departure/Arrival time convenient, Ease of Online booking, Leg room service, On-board service")
fitur = ['Flight Distance', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Leg room service', 'On-board service']
X = df[fitur]
y = df['satisfaction']
print("\nFitur yang digunakan:", list(X.columns))
print(f"Jumlah fitur telah berkurang menjadi : {X.shape[1]} kolom")

print("\nMembagi data menjadi (Data Test = 20%, dan Data Train = 80%)")
data = list(zip(X.values, y.values))
shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]
DATA_SIZE = 200
DATA_TRAIN = 8000

if len(test_data) > DATA_SIZE:
    test_data = random.sample(test_data, k=DATA_SIZE)

if len(train_data) > DATA_TRAIN:
    train_data = random.sample(train_data, k=DATA_TRAIN)
print(f"Jumlah data train: {len(train_data)}", f"Jumlah data test: {len(test_data)}")

def hitung_jarak(data1, data2):
    total = 0
    for i in range(len(data1)):
        total = total + abs(data1[i] - data2[i])
    return total

print(f"\nNilai K yang digunakan = {K}")

benar = 0
salah = 0

for data_test, kelas_asli in test_data:

    daftar_jarak = []
    for data_train, kelas_train in train_data:

        jarak = hitung_jarak(data_test, data_train)

        daftar_jarak.append([jarak, kelas_train])
    daftar_jarak.sort(key=lambda x: x[0])
    tetangga = daftar_jarak[:K]
    puas = 0
    tidak_puas = 0
    for tetangga_ke in tetangga:
        if tetangga_ke[1] == "satisfied":
            puas += 1
        else:
            tidak_puas += 1
    if puas > tidak_puas:
        prediksi = "satisfied"
    else:
        prediksi = "neutral or dissatisfied"
    if prediksi == kelas_asli:
        benar += 1
    else:
        salah += 1

total_data = benar + salah
akurasi = (benar / total_data) * 100
print("\nHasil Pengujian")
print("-"*70)
print("Data Benar :", benar)
print("Data Salah :", salah)
print(f"Akurasi : {akurasi:.2f}%")