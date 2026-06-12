# Parallel Federated Learning Framework for Heterogeneous LAN Cluster using OpenMP-based Local Training and Parallel Aggregation

Program ini adalah implementasi dari framework *Federated Learning* yang dirancang untuk berjalan pada cluster *Local Area Network* (LAN) yang heterogen. Program ini menggunakan *OpenMP* untuk memparalelkan proses *Local Training* pada masing-masing *Worker* dan proses *Parallel Aggregation* (Federated Averaging) pada *Master Node*.

## Program Description

Program ini terdiri dari tiga file:

1.  **`shared.h`**: File header yang mendefinisikan struktur data `ModelPacket` yang digunakan untuk komunikasi antara Master dan Worker. File ini juga mendefinisikan ukuran model (`MODEL_SIZE`), port jaringan (`PORT`), dan token validasi (`MAGIC_NUMBER`).
2.  **`master.cpp`**: File yang dijalankan di *Master Node* atau server agregasi. Master bertugas untuk:
    *   Menginisialisasi model global awal.
    *   Menunggu koneksi dari *Worker Node* di jaringan LAN.
    *   Mendistribusikan model global ke setiap Worker.
    *   Menerima pembaruan bobot dari setiap Worker.
    *   Melakukan agregasi paralel (FedAvg) menggunakan OpenMP untuk menggabungkan model-model lokal menjadi model global baru.
3.  **`worker.cpp`**: File yang dijalankan di *Worker Node*. Worker bertugas untuk:
    *   Menghasilkan dataset *dummy* lokal untuk pelatihan.
    *   Terhubung ke Master Node.
    *   Menerima model global dari Master.
    *   Melakukan *Heavy Local Training* (pelatihan lokal) menggunakan OpenMP untuk memparalelkan komputasi *gradient descent* pada *batch* data.
    *   Mengirimkan kembali model lokal yang telah dilatih ke Master.

## Prerequisite

Berikut prasyarat yang harus dimiliki untuk meng-compile dan menjalankan program ini :

*   Satu komputer atau *Virtual Machine* berbasis Unix/Linux yang bertindak sebagai *Master Node*.
*   Dua atau lebih komputer atau *Virtual Machine* berbasis Unix/Linux yang bertindak sebagai *Worker Node*.
*   *Compiler* C++ yang mendukung OpenMP (seperti GCC/g++) untuk setiap komputer atau *Virtual Machine*.
*   Baik Master Node maupun Worker Node harus berada dalam jaringan LAN yang sama.

## Cara Meng-Compile

Buka terminal dan jalankan *command* berikut:

```bash
# Compile di Master Node
g++ -O3 -fopenmp master.cpp -o master

# Compile di Worker Node
g++ -O3 -fopenmp worker.cpp -o worker
```

## Cara Menjalankan Program

### 1. Konfigurasi IP Master

Sebelum menjalankan Worker, pastikan sudah mengatur alamat IP Master Node dengan benar di dalam file `worker.cpp`.

Buka `worker.cpp` dan cari baris berikut :
```cpp
if (inet_pton(AF_INET, "192.168.34.253", &serv_addr.sin_addr) <= 0) {
```
Ganti `"192.168.34.253"` dengan alamat IP LAN dari komputer yang akan menjalankan `master`. Setelah diubah, compile ulang `worker.cpp`.

*By default*, `master.cpp` diatur untuk menunggu 2 koneksi Worker (`int expected_workers = 2;`). Ubah nilai ini di `master.cpp` jika ingin menggunakan jumlah Worker yang berbeda.

### 2. Jalankan Master Node

Jalankan Master Node terlebih dahulu pada satu terminal/komputer:

```bash
./master
```
Master akan mulai berjalan dan menampilkan pesan bahwa ia siap me-listen koneksi pada port 28080.

### 3. Jalankan Worker Node

Buka terminal di komputer atau *Virtual Machine* lain dalam LAN yang sama dan jalankan Worker Node :

```bash
./worker
```

## Penjelasan Output Program

### Output pada Master Node

1.  **Status Koneksi**: Master akan menampilkan pesan ketika Worker terhubung (`Worker X terhubung ke socket!`).
2.  **Penerimaan Data**: Menampilkan proses penerimaan data dari masing-masing Worker dan memvalidasi ukurannya serta `MAGIC_NUMBER`.
3.  **Proses Agregasi**: Menampilkan pesan saat memulai *Parallel Aggregation* (FedAvg) via OpenMP.
4.  **Hasil Akhir**: Menampilkan sebagian dari bobot Model Global Baru setelah agregasi selesai.
5.  **Waktu Komputasi**: Menampilkan total waktu yang dibutuhkan untuk melakukan agregasi paralel (dalam milidetik).

### Output pada Worker Node

1.  **Inisialisasi Data**: Menampilkan informasi alokasi memori untuk dataset *dummy* lokal.
2.  **Status Koneksi**: Menampilkan status saat mencoba terhubung ke Master dan saat berhasil menerima Model Global yang valid.
3.  **Proses Pelatihan (Local Training)**:
    *   Menampilkan jumlah *thread* OpenMP yang digunakan.
    *   Menampilkan progres pelatihan per *epoch* (setiap kelipatan 10).
4.  **Selesai Pelatihan**: Menampilkan waktu komputasi yang dihabiskan untuk pelatihan lokal (dalam detik).
5.  **Pengiriman Data**: Menampilkan status keberhasilan pengiriman model yang telah di-*update* kembali ke Master.

## Fitur Utama

*   **OpenMP Parallelization**: Digunakan secara ekstensif baik di Master (untuk agregasi matriks bobot yang besar secara paralel) maupun di Worker (untuk memparalelkan perhitungan gradien pada setiap *batch* data).
*   **TCP Sockets**: Menggunakan *socket* TCP standar untuk komunikasi jaringan yang andal antara Master dan Worker.
*   **Gradient & Error Clipping**: Diimplementasikan pada Worker untuk mencegah *exploding gradients* dan menjaga kestabilan numerik selama pelatihan.
*   **Magic Token Validation**: Menggunakan `MAGIC_NUMBER` untuk memastikan bahwa Master hanya memproses paket data yang valid dari Worker yang sah.
