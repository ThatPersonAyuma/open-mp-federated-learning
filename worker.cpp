#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include "shared.h"

// Variabel data global lokal
int num_samples = 0;
std::vector<float> local_X;
std::vector<float> local_y;

// Mengambil ID Worker dari Environment Variable (Misal: WORKER_ID=1)
int get_worker_id() {
    char* id_str = getenv("WORKER_ID");
    if (!id_str) {
        // Fallback jika tidak diset di Docker, default ke ID 1 (Index 0 untuk file)
        return 1; 
    }
    return std::stoi(id_str);
}

// Fungsi untuk memuat berkas dataset dinamis sesuai fase generasi berjalan
bool load_dataset_from_bin(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[DATA ERROR] Gagal membuka file dataset lokal: " << filepath << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
    if (num_samples <= 0) {
        std::cerr << "[DATA ERROR] Dataset kosong di dalam file." << std::endl;
        return false;
    }

    local_X.resize(num_samples * MODEL_SIZE);
    local_y.resize(num_samples);

    in.read(reinterpret_cast<char*>(local_X.data()), num_samples * MODEL_SIZE * sizeof(float));
    in.read(reinterpret_cast<char*>(local_y.data()), num_samples * sizeof(float));
    in.close();

    std::cout << "[DATA] Sukses memuat " << num_samples << " sampel dari " << filepath << std::endl;
    return true;
}

// Fungsi pembantu untuk mengirimkan seluruh paket biner
bool send_all(int socket, const char* data, size_t size) {
    size_t total_sent = 0;
    while (total_sent < size) {
        ssize_t bytes = send(socket, data + total_sent, size - total_sent, 0);
        if (bytes <= 0) return false;
        total_sent += bytes;
    }
    return true;
}

// Fungsi pembantu untuk menerima paket biner secara utuh
bool recv_all(int socket, char* buffer, size_t size) {
    size_t total_received = 0;
    while (total_received < size) {
        ssize_t bytes = recv(socket, buffer + total_received, size - total_received, 0);
        if (bytes <= 0) return false;
        total_received += bytes;
    }
    return true;
}

// Fungsi pembantu membuat koneksi baru ke Master (dengan penanganan Error Server Offline)
int connect_to_master(const char* host, const std::string& portStr) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    addrinfo hints{}, *result = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, portStr.c_str(), &hints, &result) != 0 || result == nullptr) {
        close(sock);
        return -1;
    }

    struct sockaddr_in serv_addr;
    memcpy(&serv_addr, result->ai_addr, sizeof(serv_addr));
    freeaddrinfo(result);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        close(sock);
        return -1; // Mengembalikan -1 jika server belum siap/offline
    }

    return sock;
}

// ====================================================================
// PERBAIKAN 1: OPTIMASI HEAVY LOCAL TRAINING (PENGHILANGAN CRITICAL BLOCK)
// ====================================================================
void train_local_model(ModelPacket &packet, int epochs, float lr) {
    std::cout << "[TRAIN] Memulai Local Training (OpenMP: " << omp_get_max_threads() << " threads)..." << std::endl;
    double start_time = omp_get_wtime();
    int batch_size = 256; 

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int b = 0; b < num_samples; b += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - b);
            
            // Step 1: Hitung semua prediksi dan error untuk batch ini terlebih dahulu (Lebih Efisien)
            std::vector<float> errors(current_batch_size, 0.0f);
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < current_batch_size; i++) {
                int sample_idx = b + i;
                int base_idx = sample_idx * MODEL_SIZE;
                float prediction = 0.0f;

                for (int k = 0; k < MODEL_SIZE; k++) {
                    prediction += local_X[base_idx + k] * packet.weights[k];
                }

                float err = prediction - local_y[sample_idx];
                
                // Error clipping tetap dipertahankan untuk stabilitas
                if (err > 5.0f) err = 5.0f;
                if (err < -5.0f) err = -5.0f;
                
                errors[i] = err;
            }

            // Step 2: Hitung akumulasi gradien tiap fitur secara paralel
            std::vector<float> batch_gradients(MODEL_SIZE, 0.0f);

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < MODEL_SIZE; j++) {
                float total_feature_grad = 0.0f;
                for (int i = 0; i < current_batch_size; i++) {
                    int sample_idx = b + i;
                    total_feature_grad += errors[i] * local_X[sample_idx * MODEL_SIZE + j];
                }
                batch_gradients[j] = total_feature_grad;
            }

            // Step 3: Update bobot dengan Gradient Clipping yang adaptif
            for (int j = 0; j < MODEL_SIZE; j++) {
                float grad = batch_gradients[j] / current_batch_size;
                
                // PERUBAHAN: Melonggarkan clipping demi mengakomodasi 4 fitur utama (70% kontribusi)
                float clip_bound = (j < 4) ? 10.0f : 2.0f; 
                
                if (grad > clip_bound) grad = clip_bound;
                if (grad < -clip_bound) grad = -clip_bound;

                packet.weights[j] -= lr * grad;
            }
        }
    }
    double end_time = omp_get_wtime();
    packet.data_size = num_samples;
    std::cout << "[TRAIN] Selesai dalam " << (end_time - start_time) << " detik." << std::endl;
}
// void train_local_model(ModelPacket &packet, int epochs, float lr) {
//     std::cout << "[TRAIN] Memulai Local Training (OpenMP: " << omp_get_max_threads() << " threads)..." << std::endl;
//     double start_time = omp_get_wtime();
//     int batch_size = 256; 

//     for (int epoch = 0; epoch < epochs; epoch++) {
//         for (int b = 0; b < num_samples; b += batch_size) {
//             int current_batch_size = std::min(batch_size, num_samples - b);
            
//             // Menggunakan alokasi flat zero untuk menampung gradien batch
//             std::vector<float> batch_gradients(MODEL_SIZE, 0.0f);

//             // Paralelisasi di tingkat kalkulasi fitur tanpa blocking critical
//             #pragma omp parallel for schedule(static)
//             for (int j = 0; j < MODEL_SIZE; j++) {
//                 float total_feature_grad = 0.0f;

//                 for (int i = b; i < b + current_batch_size; i++) {
//                     float prediction = 0.0f;
//                     int base_idx = i * MODEL_SIZE;

//                     // Unrolling internal / SIMD teroptimasi otomatis oleh compiler
//                     for (int k = 0; k < MODEL_SIZE; k++) {
//                         prediction += local_X[base_idx + k] * packet.weights[k];
//                     }

//                     float error = prediction - local_y[i];
//                     if (error > 5.0f) error = 5.0f;
//                     if (error < -5.0f) error = -5.0f;

//                     total_feature_grad += error * local_X[base_idx + j];
//                 }
                
//                 // Langsung simpan ke indeks j tanpa rebutan antar thread
//                 batch_gradients[j] = total_feature_grad;
//             }

//             // Update bobot + Gradient Clipping
//             for (int j = 0; j < MODEL_SIZE; j++) {
//                 float grad = batch_gradients[j] / current_batch_size;
//                 if (grad > 1.0f) grad = 1.0f;
//                 if (grad < -1.0f) grad = -1.0f;

//                 packet.weights[j] -= lr * grad;
//             }
//         }
//     }
//     double end_time = omp_get_wtime();
//     packet.data_size = num_samples;
//     std::cout << "[TRAIN] Selesai dalam " << (end_time - start_time) << " detik." << std::endl;
// }

// Fungsi Heavy Local Training via OpenMP (Sama seperti logika dasar Anda)
// void train_local_model(ModelPacket &packet, int epochs, float lr) {
//     std::cout << "[TRAIN] Memulai Local Training (OpenMP: " << omp_get_max_threads() << " threads)..." << std::endl;
//     double start_time = omp_get_wtime();
//     int batch_size = 256; 

//     for (int epoch = 0; epoch < epochs; epoch++) {
//         for (int b = 0; b < num_samples; b += batch_size) {
//             int current_batch_size = std::min(batch_size, num_samples - b);
//             std::vector<float> batch_gradients(MODEL_SIZE, 0.0f);

//             #pragma omp parallel
//             {
//                 std::vector<float> local_gradients(MODEL_SIZE, 0.0f);
//                 #pragma omp for schedule(static)
//                 for (int i = b; i < b + current_batch_size; i++) {
//                     float prediction = 0.0f;
//                     int base_idx = i * MODEL_SIZE;

//                     for (int j = 0; j < MODEL_SIZE; j++) {
//                         prediction += local_X[base_idx + j] * packet.weights[j];
//                     }

//                     float error = prediction - local_y[i];
//                     if (error > 5.0f) error = 5.0f;
//                     if (error < -5.0f) error = -5.0f;

//                     for (int j = 0; j < MODEL_SIZE; j++) {
//                         local_gradients[j] += error * local_X[base_idx + j];
//                     }
//                 }

//                 #pragma omp critical
//                 {
//                     for (int j = 0; j < MODEL_SIZE; j++) {
//                         batch_gradients[j] += local_gradients[j];
//                     }
//                 }
//             }

//             for (int j = 0; j < MODEL_SIZE; j++) {
//                 float grad = batch_gradients[j] / current_batch_size;
//                 if (grad > 1.0f) grad = 1.0f;
//                 if (grad < -1.0f) grad = -1.0f;

//                 packet.weights[j] -= lr * grad;
//             }
//         }
//     }
//     double end_time = omp_get_wtime();
//     packet.data_size = num_samples;
//     std::cout << "[TRAIN] Selesai dalam " << (end_time - start_time) << " detik." << std::endl;
// }

int main() {
    char* host = getenv("MASTER_HOST");
    char* portEnv = getenv("MASTER_PORT");

    if (!host || !portEnv) {
        std::cerr << "Variabel lingkungan MASTER_HOST atau MASTER_PORT belum diset!\n";
        return -1;
    }
    std::string port_str(portEnv);
    int worker_id = get_worker_id();

    int local_generation_tracker = -1; // Penanda internal siklus terakhir yang dikerjakan
    int successful_updates_count = 0;   // Penghitung jumlah update sukses hingga target 10 kali

    std::cout << "=========================================================" << std::endl;
    std::cout << "Worker ID: " << worker_id << " Online. Memulai Polling ke Master..." << std::endl;
    std::cout << "=========================================================" << std::endl;

    // --- LOOP UTAMA BERBASIS POLLING SIKLUS (MAKSIMAL 10 KALI UPDATE BERHASIL) ---
    while (successful_updates_count < 10) {
        
        // 1. Coba hubungi Master untuk menanyakan Info Generasi Terkini
        int sock = connect_to_master(host, port_str);
        if (sock < 0) {
            std::cout << "[POLLING] Master belum siap / offline. Mencoba kembali dalam 5 detik..." << std::endl;
            sleep(5);
            continue;
        }

        // Kirim Request Token INFO
        ModelPacket req_packet{};
        req_packet.magic_token = MAGIC_NUMBER;
        req_packet.command = CommandType::REQ_GENERATION_INFO;

        if (!send_all(sock, (char*)&req_packet, sizeof(ModelPacket))) {
            close(sock);
            sleep(5);
            continue;
        }

        // Terima Balasan Model dan Info Generasi dari Master
        ModelPacket incoming_model;
        if (!recv_all(sock, (char*)&incoming_model, sizeof(ModelPacket)) || incoming_model.magic_token != MAGIC_NUMBER) {
            std::cerr << "[ERROR] Gagal menerima info model global dari master." << std::endl;
            close(sock);
            sleep(5);
            continue;
        }
        close(sock); // Tutup koneksi endpoint request info secepatnya

        int master_generation = incoming_model.current_generation;

        // 2. CEK JIKA GENERASI SAMA DENGAN YANG PERNAH DIKERJAKAN
        if (master_generation == local_generation_tracker) {
            std::cout << "[POLLING] Master masih berada di Generation " << master_generation 
                      << ". Menunggu update dari worker lain (sisa target: " << (10 - successful_updates_count) << ")..." << std::endl;
            sleep(5);
            continue;
        }

        // Jika ada perubahan generasi di Master, lakukan sinkronisasi eksekusi
        std::cout << "\n[NEW GENERATION] Terdeteksi transisi ke Generation: " << master_generation << std::endl;

        // 3. SELEKSI FILE DATASET SESUAI ATURAN (5 Kali Pertama Past, 5 Kali Berikutnya Now)
        std::string current_dataset_path;
        int file_index = worker_id - 1; // Sesuai instruksi {ID-1}

        if (successful_updates_count < 5) {
            current_dataset_path = "data/dataset_past_worker_" + std::to_string(file_index) + ".bin";
        } else {
            current_dataset_path = "data/dataset_now_worker_" + std::to_string(file_index) + ".bin";
        }

        // Muat dataset dinamis
        if (!load_dataset_from_bin(current_dataset_path)) {
            std::cerr << "[SKIP] Melewati generasi karena kegagalan muat data lokal. Coba lagi dalam 5 detik..." << std::endl;
            sleep(5);
            continue;
        }

        // 4. JALANKAN TRAINING BERDASARKAN MODEL TERBARU MASTER
        train_local_model(incoming_model, 10, 0.001f);

        // 5. KIRIM HASIL UPDATE BOBOT KEMBALI KE MASTER (ENDPOINT: SEND_WORKER_UPDATE)
        int write_sock = connect_to_master(host, port_str);
        if (write_sock < 0) {
            std::cerr << "[CONNECTION LOST] Gagal mengirim setoran karena Master tiba-tiba offline!" << std::endl;
            sleep(5);
            continue;
        }

        // Konfigurasi paket pengiriman update
        incoming_model.command = CommandType::SEND_WORKER_UPDATE;
        incoming_model.current_generation = master_generation; // Berikan tanda pengenal generasi berjalan

        if (send_all(write_sock, (char*)&incoming_model, sizeof(ModelPacket))) {
            std::cout << "[SUCCESS] Update untuk Generation " << master_generation << " sukses terkirim." << std::endl;
            
            // Perbarui tracker internal agar tidak memproses generasi yang sama lagi
            local_generation_tracker = master_generation;
            successful_updates_count++;
        } else {
            std::cerr << "[ERROR] Putus koneksi saat mengirim data update ke master." << std::endl;
        }

        close(write_sock);
        sleep(5); // Jeda aman antar siklus kerja
    }

    std::cout << "\n=========================================================" << std::endl;
    std::cout << " Selesai! Worker telah berhasil menyelesaikan 10 siklus update." << std::endl;
    std::cout << "=========================================================" << std::endl;

    return 0;
}
// #include <iostream>
// #include <vector>
// #include <sys/socket.h>
// #include <netdb.h>
// #include <arpa/inet.h>
// #include <unistd.h>
// #include <omp.h>
// #include <cstring>
// #include <cmath>
// #include <cstdlib>
// #include <fstream>
// #include "shared.h"

// int num_samples = 20000;

// std::vector<float> local_X;
// std::vector<float> local_y;

// void initialize_dummy_data() {
//     std::cout << "Mengalokasikan " << (num_samples * MODEL_SIZE * 4.0) / (1024*1024) << " MB RAM untuk dataset Rekomendasi Pelanggan..." << std::endl;
//     local_X.resize(num_samples * MODEL_SIZE);
//     local_y.resize(num_samples);

//     // Seed untuk randomisasi
//     srand(12345);

//     for(int i = 0; i < num_samples; i++) {
//         float target_score = 0.0f;

//         for(int j = 0; j < MODEL_SIZE; j++) {
//             float feature_val = 0.0f;
            
//             // Simulasi fitur pelanggan
//             if (j == 0) {
//                 // Fitur 0: Umur (dinormalisasi 0-1, misal 18-60 tahun)
//                 feature_val = (rand() % 43 + 18) / 60.0f;
//             } else if (j == 1) {
//                 // Fitur 1: Waktu browsing di aplikasi (menit, dinormalisasi max 120 menit)
//                 feature_val = (rand() % 121) / 120.0f;
//             } else if (j == 2) {
//                 // Fitur 2: Jumlah transaksi sebelumnya (dinormalisasi max 50 transaksi)
//                 feature_val = (rand() % 51) / 50.0f;
//             } else if (j == 3) {
//                 // Fitur 3: Rating rata-rata yang diberikan pelanggan (0-5, dinormalisasi 0-1)
//                 feature_val = (rand() % 51) / 50.0f;
//             } else {
//                 // Fitur lainnya: Data historis interaksi produk, klik, preferensi kategori, dll.
//                 // Menggunakan pola pseudo-random yang lebih ringan komputasinya
//                 feature_val = (float)((i + j) % 100) * 0.01f;
//             }
            
//             local_X[i * MODEL_SIZE + j] = feature_val;

//             // Buat target score (probabilitas rekomendasi/pembelian) bergantung pada fitur utama
//             if (j < 4) {
//                 target_score += feature_val * 0.25f; // Bobot rata untuk 4 fitur utama
//             }
//         }
        
//         // Tambahkan sedikit noise pada target untuk realisme
//         float noise = ((rand() % 21) - 10) * 0.01f; // -0.1 hingga 0.1
//         target_score += noise;
        
//         // Batasi target antara 0.0 dan 1.0
//         if (target_score > 1.0f) target_score = 1.0f;
//         if (target_score < 0.0f) target_score = 0.0f;

//         local_y[i] = target_score;
//     }
//     std::cout << "Inisialisasi dataset Rekomendasi Pelanggan selesai.\n" << std::endl;
// }

// void train_local_model(ModelPacket &packet, int epochs, float lr) {
//     std::cout << "Mulai Heavy Local Training (OpenMP: " << omp_get_max_threads() << " threads)..." << std::endl;
//     double start_time = omp_get_wtime();

//     int batch_size = 256; 

//     for (int epoch = 0; epoch < epochs; epoch++) {
//         for (int b = 0; b < num_samples; b += batch_size) {
//             int current_batch_size = std::min(batch_size, num_samples - b);
//             std::vector<float> batch_gradients(MODEL_SIZE, 0.0f);

//             #pragma omp parallel
//             {
//                 std::vector<float> local_gradients(MODEL_SIZE, 0.0f);

//                 #pragma omp for schedule(static)
//                 for (int i = b; i < b + current_batch_size; i++) {
//                     float prediction = 0.0f;
//                     int base_idx = i * MODEL_SIZE;

//                     for (int j = 0; j < MODEL_SIZE; j++) {
//                         prediction += local_X[base_idx + j] * packet.weights[j];
//                     }

//                     float error = prediction - local_y[i];

//                     // 1. Error Clipping (Mencegah prediksi liar)
//                     if (error > 5.0f) error = 5.0f;
//                     if (error < -5.0f) error = -5.0f;

//                     for (int j = 0; j < MODEL_SIZE; j++) {
//                         local_gradients[j] += error * local_X[base_idx + j];
//                     }
//                 }

//                 #pragma omp critical
//                 {
//                     for (int j = 0; j < MODEL_SIZE; j++) {
//                         batch_gradients[j] += local_gradients[j];
//                     }
//                 }
//             }

//             // Update bobot + 2. Gradient Clipping
//             for (int j = 0; j < MODEL_SIZE; j++) {
//                 float grad = batch_gradients[j] / current_batch_size;
                
//                 // Batasi gradien maksimal demi kestabilan matematika
//                 if (grad > 1.0f) grad = 1.0f;
//                 if (grad < -1.0f) grad = -1.0f;

//                 packet.weights[j] -= lr * grad;
//             }
//         }

//         // Cek deteksi NaN
//         if (std::isnan(packet.weights[0])) {
//             std::cerr << "Kritis: Bobot mendeteksi NaN di Epoch " << epoch + 1 << "!" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         if ((epoch + 1) % 10 == 0) {
//             std::cout << "  -> Progress: Epoch " << epoch + 1 << "/" << epochs << " selesai." << std::endl;
//         }
//     }

//     double end_time = omp_get_wtime();
//     packet.data_size = num_samples;
//     std::cout << "\n[!] Local Training Selesai!" << std::endl;
//     std::cout << "[!] Waktu Komputasi Worker: " << (end_time - start_time) << " detik." << std::endl;
// }

// int main() {
//     char* host = getenv("MASTER_HOST");
//     char* portEnv = getenv("MASTER_PORT");

//     std::cin;
//     std::cout
//         << "Connecting to "
//         << host
//         << ":"
//         << portEnv
//         << std::endl;
//     initialize_dummy_data();
//     int sock = 0;
//     struct sockaddr_in serv_addr;
//     ModelPacket packet;

//     if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
//         std::cerr << "Socket creation error" << std::endl;
//         return -1;
//     }

//     serv_addr.sin_family = AF_INET;
//     int port = std::stoi(portEnv);
//     serv_addr.sin_port =  htons(port);

//     const char* masterHost = getenv("MASTER_HOST");


//     if(masterHost == nullptr)
//     {
//         std::cerr << "MASTER_HOST not set\n";
//         return -1;
//     }


//     std::string portStr = std::to_string(port);

//     addrinfo hints{};
//     addrinfo* result = nullptr;

//     hints.ai_family = AF_INET;
//     hints.ai_socktype = SOCK_STREAM;

//     int status = getaddrinfo(
//         masterHost,
//         portStr.c_str(),
//         &hints,
//         &result
//     );


//     if(status != 0 || result == nullptr)
//     {
//         std::cerr 
//             << "Host not found: "
//             << masterHost
//             << std::endl;

//         return -1;
//     }

//     sockaddr_in* addr =
//         reinterpret_cast<sockaddr_in*>(result->ai_addr);

//     memcpy(
//         &serv_addr.sin_addr,
//         &addr->sin_addr,
//         sizeof(addr->sin_addr)
//     );

//     std::cout << "Menghubungkan ke Master..." << std::endl;
//     if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
//         std::cerr << "Connection Failed" << std::endl;
//         return -1;
//     }

//     // Terima & Validasi
//     int total_received = 0;
//     char* recv_ptr = (char*)&packet;
//     while (total_received < sizeof(ModelPacket)) {
//         int bytes = recv(sock, recv_ptr + total_received, sizeof(ModelPacket) - total_received, 0);
//         if (bytes <= 0) {
//             std::cerr << "Gagal menerima data dari Master" << std::endl;
//             break;
//         }
//         total_received += bytes;
//     }

//     if (packet.magic_token == MAGIC_NUMBER) {
//         std::cout << "Model Global (Valid) diterima dari Master." << std::endl;
//     }

//     // Eksekusi Training (50 Epochs)
//     train_local_model(packet, 50, 0.001f);

//     // Kirim Balik
//     unsigned long total_sent = 0;
//     char* send_ptr = (char*)&packet;
//     while (total_sent < sizeof(ModelPacket)) {
//         int bytes = send(sock, send_ptr + total_sent, sizeof(ModelPacket) - total_sent, 0);
//         if (bytes <= 0) {
//             std::cerr << "Gagal mengirim data ke Master" << std::endl;
//             break;
//         }
//         total_sent += bytes;
//     }
//     std::cout << "Update bobot berhasil dikirim ke Master." << std::endl;

//     shutdown(sock, SHUT_WR);
//     close(sock);
//     return 0;
// }