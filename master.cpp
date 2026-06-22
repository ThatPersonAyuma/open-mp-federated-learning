#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <fstream>
#include <csignal>
#include <sys/stat.h>
#include "shared.h"

volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    std::cout << "\n[SIGNAL] Menerima signal (" << signum << "), mempersiapkan shutdown master..." << std::endl;
    keep_running = 0;
}

// ====================================================================
// PERBAIKAN 2: PARALELISASI INFERENCE DI EVALUASI MODEL
// ====================================================================
bool evaluate_model(const std::string& filepath, const float* weights) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[EVALUASI ERROR] Gagal membuka file testing: " << filepath << std::endl;
        return false;
    }

    int num_samples = 0;
    in.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
    if (num_samples <= 0) {
        std::cerr << "[EVALUASI ERROR] Sampel testing kosong atau tidak valid." << std::endl;
        return false;
    }

    // Mengalokasikan seluruh buffer memori testing di awal (Lebih aman untuk OpenMP)
    std::vector<float> all_features(num_samples * MODEL_SIZE);
    std::vector<float> true_labels(num_samples);
    std::vector<float> pred_labels(num_samples);

    in.read(reinterpret_cast<char*>(all_features.data()), num_samples * MODEL_SIZE * sizeof(float));
    in.read(reinterpret_cast<char*>(true_labels.data()), num_samples * sizeof(float));
    in.close();

    // Jalankan kalkulasi dot-product prediksi secara paralel penuh via OpenMP
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < num_samples; s++) {
        float prediction = 0.0f;
        int base_idx = s * MODEL_SIZE;

        for (int i = 0; i < MODEL_SIZE; i++) {
            prediction += all_features[base_idx + i] * weights[i];
        }

        // Biarkan nilai murni apa adanya tanpa kliping paksa agar R2 score akurat
        pred_labels[s] = prediction;
    }

    // Hitung Rata-rata Label Aktual (untuk R-squared)
    double sum_true = 0.0;
    #pragma omp parallel for reduction(+:sum_true)
    for (int i = 0; i < num_samples; i++) {
        sum_true += true_labels[i];
    }
    double mean_true = sum_true / num_samples;

    // Hitung Metrik Evaluasi Regresi Komparatif via OpenMP
    double ss_residual = 0.0; 
    double ss_total = 0.0;    
    double absolute_error_sum = 0.0;

    #pragma omp parallel for reduction(+:ss_residual, ss_total, absolute_error_sum)
    for (int i = 0; i < num_samples; i++) {
        double diff = true_labels[i] - pred_labels[i];
        ss_residual += diff * diff;
        absolute_error_sum += std::abs(diff);

        double dev = true_labels[i] - mean_true;
        ss_total += dev * dev;
    }

    double mse = ss_residual / num_samples;
    double rmse = std::sqrt(mse);
    double mae = absolute_error_sum / num_samples;
    double r_squared = 1.0 - (ss_residual / (ss_total + 1e-10)); 

    std::cout << "\n=========================================================" << std::endl;
    std::cout << "        LAPORAN EVALUASI MODEL (REGRESI UNBIASED)        " << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << " Mean Absolute Error (MAE)      : " << mae << std::endl;
    std::cout << " Mean Squared Error (MSE)       : " << mse << std::endl;
    std::cout << " Root Mean Squared Error (RMSE) : " << rmse << std::endl;
    std::cout << " R-squared (R2 Score)           : " << r_squared << " (" << r_squared * 100.0 << "%)" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    return true;
}

// --- FUNGSI TESTING & EVALUASI REGRESI ---
// bool evaluate_model(const std::string& filepath, const float* weights) {
//     std::ifstream in(filepath, std::ios::binary);
//     if (!in.is_open()) {
//         std::cerr << "[EVALUASI ERROR] Gagal membuka file testing: " << filepath << std::endl;
//         return false;
//     }

//     int num_samples = 0;
//     in.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
//     if (num_samples <= 0) {
//         std::cerr << "[EVALUASI ERROR] Sampel testing kosong atau tidak valid." << std::endl;
//         return false;
//     }

//     std::vector<float> true_labels(num_samples);
//     std::vector<float> pred_labels(num_samples);

//     // 1. Load Data dan Lakukan Inference/Prediksi
//     for (int s = 0; s < num_samples; s++) {
//         std::vector<float> features(MODEL_SIZE);
//         in.read(reinterpret_cast<char*>(features.data()), MODEL_SIZE * sizeof(float));
//         in.read(reinterpret_cast<char*>(&true_labels[s]), sizeof(float));

//         // Prediksi linear (Dot Product)
//         float prediction = 0.0f;
//         for (int i = 0; i < MODEL_SIZE; i++) {
//             prediction += features[i] * weights[i];
//         }

//         // Batasi hasil prediksi sesuai rentang probabilitas 0 s.d 1
//         if (prediction > 1.0f) prediction = 1.0f;
//         if (prediction < 0.0f) prediction = 0.0f;

//         pred_labels[s] = prediction;
//     }
//     in.close();

//     // 2. Hitung Rata-rata Label Aktual (untuk R-squared)
//     double sum_true = 0.0;
//     #pragma omp parallel for reduction(+:sum_true)
//     for (int i = 0; i < num_samples; i++) {
//         sum_true += true_labels[i];
//     }
//     double mean_true = sum_true / num_samples;

//     // 3. Hitung Metrik Evaluasi Regresi Komparatif via OpenMP
//     double ss_residual = 0.0; // Sum of Squared Residuals
//     double ss_total = 0.0;    // Total Sum of Squares
//     double absolute_error_sum = 0.0;

//     #pragma omp parallel for reduction(+:ss_residual, ss_total, absolute_error_sum)
//     for (int i = 0; i < num_samples; i++) {
//         double diff = true_labels[i] - pred_labels[i];
//         ss_residual += diff * diff;
//         absolute_error_sum += std::abs(diff);

//         double dev = true_labels[i] - mean_true;
//         ss_total += dev * dev;
//     }

//     double mse = ss_residual / num_samples;
//     double rmse = std::sqrt(mse);
//     double mae = absolute_error_sum / num_samples;
//     double r_squared = 1.0 - (ss_residual / (ss_total + 1e-10)); // Tambah epsilon kecil anti-div-by-zero

//     // 4. Print Laporan Hasil Evaluasi
//     std::cout << "\n=========================================================" << std::endl;
//     std::cout << "        LAPORAN EVALUASI MODEL (REGRESI)                 " << std::endl;
//     std::cout << "=========================================================" << std::endl;
//     std::cout << " File Path       : " << filepath << std::endl;
//     std::cout << " Jumlah Sampel   : " << num_samples << " pelanggan" << std::endl;
//     std::cout << "---------------------------------------------------------" << std::endl;
//     std::cout << " Mean Absolute Error (MAE)      : " << mae << std::endl;
//     std::cout << " Mean Squared Error (MSE)       : " << mse << std::endl;
//     std::cout << " Root Mean Squared Error (RMSE) : " << rmse << std::endl;
//     std::cout << " R-squared (R2 Score)           : " << r_squared << " (" << r_squared * 100.0 << "%)" << std::endl;
//     std::cout << "=========================================================\n" << std::endl;

//     return true;
// }

bool send_all(int socket, const char* data, size_t size) {
    size_t total_sent = 0;
    while (total_sent < size) {
        ssize_t bytes = send(socket, data + total_sent, size - total_sent, 0);
        if (bytes <= 0) return false;
        total_sent += bytes;
    }
    return true;
}

int main() {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    const char* char_ew = getenv("EXPECTED_WORKER");
    if (!char_ew) {
        std::cerr << "EXPECTED_WORKER environment variable tidak ditemukan!\n";
        return -1;
    }
    const int expected_workers = std::stoi(char_ew);

    int current_generation = 0;

    ModelPacket global_model;
    global_model.magic_token = MAGIC_NUMBER;
    global_model.current_generation = current_generation;
    for (int i = 0; i < MODEL_SIZE; i++) {
        global_model.weights[i] = 0.01f; 
    }

    std::vector<ModelPacket> worker_updates;

    int server_fd;
    struct sockaddr_in address;
    int opt = 1;

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket pembuatan gagal");
        return -1;
    }

    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    int rcvbuf_size = 8 * 1024 * 1024; 
    setsockopt(server_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; 
    uint16_t PORT = static_cast<uint16_t>(getPort());
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind gagal");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, expected_workers * 2) < 0) {
        perror("Listen gagal");
        close(server_fd);
        return -1;
    }

    std::cout << "Master Node Online di Port " << PORT << "..." << std::endl;

    while (keep_running) {
        std::cout << "\n[GENERATION TRACK] Memulai Siklus Generasi: " << current_generation << std::endl;
        
        // Menggunakan path file testing sesuai instruksi Anda ("data/dataset_test_X.bin")
        // Catatan: Gunakan '/' untuk Linux/Docker (kompatibel silang dengan Windows OS)
        std::string test_file_path;
        if (current_generation<5){
            test_file_path =  "data/dataset_test_past.bin";
        }else{
            test_file_path =  "data/dataset_test_now.bin";
        }
        
        // JALANKAN PROSES EVALUASI TESTING DIAWAL SIKLUS LOOP
        std::cout << "[TESTING] Menjalankan validasi performa model global awal..." << std::endl;
        if (!evaluate_model(test_file_path, global_model.weights)) {
            std::cerr << "[WARNING] Skip siklus loop karena berkas testing generasi ke-" 
                      << current_generation << " bermasalah/tidak ditemukan. Menunda 5 detik..." << std::endl;
            sleep(5);
            continue; 
        }

        std::cout << "[STATUS] Evaluasi berhasil. Menanti koneksi masuk dari Worker..." << std::endl;
        
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(server_fd, (sockaddr*)&client_addr, &client_len);

        if (client_socket < 0) {
            if (!keep_running) break;
            perror("Accept bermasalah");
            continue;
        }

        ModelPacket incoming_packet;
        int total_bytes = 0;
        int packet_size = sizeof(ModelPacket);
        char* buffer_ptr = (char*)&incoming_packet;
        bool connection_healthy = true;

        while (total_bytes < packet_size) {
            int bytes_read = recv(client_socket, buffer_ptr + total_bytes, packet_size - total_bytes, 0);
            if (bytes_read <= 0) {
                connection_healthy = false;
                break;
            }
            total_bytes += bytes_read;
        }

        if (!connection_healthy || incoming_packet.magic_token != MAGIC_NUMBER) {
            std::cerr << "[MALFORMED] Paket asing/rusak diabaikan." << std::endl;
            close(client_socket);
            continue;
        }

        switch (incoming_packet.command) {
            case CommandType::REQ_GENERATION_INFO: {
                ModelPacket response_packet = global_model;
                response_packet.command = CommandType::REQ_GENERATION_INFO;
                response_packet.current_generation = current_generation;
                send_all(client_socket, (char*)&response_packet, sizeof(ModelPacket));
                break;
            }

            case CommandType::SEND_WORKER_UPDATE: {
                if (incoming_packet.current_generation == current_generation) {
                    worker_updates.push_back(incoming_packet);
                    std::cout << "[LOG] Sukses menampung update (" << worker_updates.size() 
                              << "/" << expected_workers << ") dari worker." << std::endl;
                } else {
                    std::cerr << "[OUTDATED] Update worker ditolak (Worker Gen: " 
                              << incoming_packet.current_generation << ", Master Gen: " << current_generation << ")" << std::endl;
                }
                break;
            }
            default:
                break;
        }

        close(client_socket);

        // JIKA KUOTA UPDATE WORKER TERPENUHI, LAKUKAN AGREGASI
        if (worker_updates.size() >= static_cast<size_t>(expected_workers)) {
            std::cout << "\n--- Memulai Agregasi Model Global Baru via OpenMP ---" << std::endl;

            int total_all_data = 0;
            for (const auto& update : worker_updates) {
                total_all_data += update.data_size;
            }

            if (total_all_data > 0) {
                double start_time = omp_get_wtime();
                std::vector<float> new_global_weights(MODEL_SIZE, 0.0f);

                #pragma omp parallel for shared(worker_updates, new_global_weights, total_all_data) schedule(static)
                for (int j = 0; j < MODEL_SIZE; j++) {
                    float sum_weight = 0.0f;
                    for (int i = 0; i < expected_workers; i++) {
                        if (!std::isnan(worker_updates[i].weights[j])) {
                            sum_weight += ((float)worker_updates[i].data_size / total_all_data) * worker_updates[i].weights[j];
                        }
                    }
                    new_global_weights[j] = sum_weight;
                }

                for (int j = 0; j < MODEL_SIZE; j++) {
                    global_model.weights[j] = new_global_weights[j];
                }

                double end_time = omp_get_wtime();
                std::cout << "[SUCCESS] Agregasi selesai dalam: " << (end_time - start_time) * 1000 << " ms" << std::endl;

                std::ofstream out("global_model.bin", std::ios::binary);
                if (out.is_open()) {
                    out.write(reinterpret_cast<char*>(global_model.weights), MODEL_SIZE * sizeof(float));
                    out.close();
                }
                
                // NAIKKAN GENERASI SIKLUS BERIKUTNYA
                current_generation++;
                global_model.current_generation = current_generation;

            } else {
                std::cerr << "[ERROR] Data size total dari seluruh worker bernilai 0." << std::endl;
            }

            worker_updates.clear();
        }
    }

    std::cout << "[SHUTDOWN] Menutup master socket server fd. Selesai." << std::endl;
    close(server_fd);
    return 0;
}
// #include <iostream>
// #include <vector>
// #include <sys/socket.h>
// #include <netinet/in.h>
// #include <unistd.h>
// #include <omp.h>
// #include <cstring>
// #include <cmath>
// #include <fstream>
// #include "shared.h"

// int GENERATION = 0; // Define so we how much model have changed, also for Identity

// int main() {
//     int server_fd, new_socket;
//     struct sockaddr_in address;
//     int opt = 1;
//     int addrlen = sizeof(address);
    
//     const char* char_ew = getenv("EXPECTED_WORKER");

//     if(!char_ew)
//     {
//         std::cerr << "EXPECTED_WORKER not set\n";
//         return -1;
//     }

//     const int expected_workers = std::stoi(char_ew);
//     // int expected_workers = 2; // Menentukan berapa client LAN yang gabung
//     std::vector<ModelPacket> worker_updates(expected_workers);
//     for(int i = 0; i < expected_workers; i++) {
//         memset(&worker_updates[i], 0, sizeof(ModelPacket));
//     }

//     // Inisialisasi Model Global Awal
//     ModelPacket global_model;
//     global_model.magic_token = MAGIC_NUMBER;
//     for(int i = 0; i < MODEL_SIZE; i++) {
//         global_model.weights[i] = 0.01f;
//     }

//     if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
//         perror("socket failed");
//         exit(EXIT_FAILURE);
//     }

//     if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
//         perror("setsockopt");
//         exit(EXIT_FAILURE);
//     }

//     int rcvbuf_size = 4 * 1024 * 1024;
//     setsockopt(server_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));
    
//     address.sin_family = AF_INET;
//     address.sin_addr.s_addr = INADDR_ANY; // Menerima koneksi dari mana saja di LAN
//     uint16_t PORT = static_cast<uint16_t>(getPort());
//     address.sin_port = htons(PORT);

//     if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
//         perror("bind failed");
//         exit(EXIT_FAILURE);
//     }
    
//     if (listen(server_fd, expected_workers) < 0) {
//         perror("listen");
//         exit(EXIT_FAILURE);
//     }

//     std::cout << "Master Node siap me-listen koneksi LAN pada port " << PORT << "..." << std::endl;

//     std::vector<int> client_sockets;
//     struct sockaddr_in client_addr;
//     socklen_t client_len = sizeof(client_addr);
//     for(int i = 0; i < expected_workers; i++) {
        
//         if ((new_socket = accept(
//             server_fd,
//             (sockaddr*)&client_addr,
//             &client_len
//         )) < 0) {
//             perror("accept");
//             exit(EXIT_FAILURE);
//         }
//         std::cout << "Worker " << i + 1 << " terhubung ke socket!" << std::endl;
//         client_sockets.push_back(new_socket);

//         // Langsung broadcast model global awal ke worker ini
//         unsigned long total_sent = 0;
//         char* send_ptr = (char*)&global_model;
//         while (total_sent < sizeof(ModelPacket)) {
//             int bytes = send(new_socket, send_ptr + total_sent, sizeof(ModelPacket) - total_sent, 0);
//             if (bytes <= 0) {
//                 std::cerr << "Gagal mengirim model ke Worker " << i + 1 << std::endl;
//                 break;
//             }
//             total_sent += bytes;
//         }
//     }

//     std::vector<int> bytes_received(expected_workers, 0);

//     #pragma omp parallel for num_threads(expected_workers)
//     for(int i = 0; i < expected_workers; i++) {
//         int total_bytes_received = 0;
//         int packet_size = sizeof(ModelPacket);
//         char* packet_buffer = (char*)&worker_updates[i];

//         std::cout << "Menerima data dari Worker " << i + 1 << "..." << std::endl;

//         while (total_bytes_received < packet_size) {
//             int bytes_read = recv(client_sockets[i], 
//                                   packet_buffer + total_bytes_received, 
//                                   packet_size - total_bytes_received, 
//                                   0);
            
//             if (bytes_read < 0) {
//                 perror("recv failed");
//                 break;
//             } else if (bytes_read == 0) {
//                 break;
//             }
            
//             total_bytes_received += bytes_read;
//         }

//         bytes_received[i] = total_bytes_received;
//     }

//     for(int i = 0; i < expected_workers; i++) {
//         if (bytes_received[i] < (int)sizeof(ModelPacket)) {
//             std::cerr << "Error: Paket data dari Worker " << i + 1 << " korup atau tidak lengkap!" << std::endl;
//             worker_updates[i].data_size = 0;
//         } else if (worker_updates[i].magic_token != MAGIC_NUMBER) {
//             std::cerr << "Peringatan: Koneksi asing terdeteksi di Worker " << i + 1 
//                       << "! Data diabaikan." << std::endl;
//             worker_updates[i].data_size = 0; 
//         } else {
//             std::cout << "Menerima update VALID dari Worker " << i + 1 
//                       << ". Data size: " << worker_updates[i].data_size << std::endl;
//         }
//         close(client_sockets[i]);
//     }

//     // --- PROSES PARALLEL AGGREGATION (FEDERATED AVERAGING) VIA OPENMP ---
//     std::cout << "Melakukan Parallel Aggregation (FedAvg) via OpenMP..." << std::endl;
    
//     int total_all_data = 0;
//     for(int i = 0; i < expected_workers; i++) {
//         total_all_data += worker_updates[i].data_size;
//     }

//     // Antisipasi jika semua koneksi ternyata invalid (total_all_data == 0)
//     if (total_all_data == 0) {
//         std::cerr << "Error: Tidak ada data valid dari worker untuk diagregasi!" << std::endl;
//         close(server_fd);
//         return -1;
//     }

//     double start_time = omp_get_wtime();

//     std::vector<float> new_global_weights(MODEL_SIZE, 0.0f);

//     // Paralelisasi kalkulasi agregasi matriks bobot yang besar
//     #pragma omp parallel for shared(worker_updates, new_global_weights, total_all_data) schedule(static)
//     for (int j = 0; j < MODEL_SIZE; j++) {
//         float sum_weight = 0.0f;
//         for (int i = 0; i < expected_workers; i++) {
//             // Melakukan pengecekan ganda: ukuran data harus valid DAN nilai tidak boleh NaN
//             if (worker_updates[i].data_size > 0 && !std::isnan(worker_updates[i].weights[j])) {
//                 sum_weight += ((float)worker_updates[i].data_size / total_all_data) * worker_updates[i].weights[j];
//             }
//         }
//         new_global_weights[j] = sum_weight;
//     }

//     double end_time = omp_get_wtime();

//     // Terapkan hasil agregasi kembali ke global model
//     std::cout << "\nHasil Model Global Baru setelah Agregasi:" << std::endl;
//     for(int j = 0; j < MODEL_SIZE; j++) {
//         global_model.weights[j] = new_global_weights[j];
//         std::cout << "W[" << j << "] = " << global_model.weights[j] << " | ";
//     }
//     std::cout << "\nWaktu komputasi Agregasi Paralel: " << (end_time - start_time) * 1000 << " ms" << std::endl;

//     // Simpan model global untuk inference
//     std::ofstream out("model/global_model.bin", std::ios::binary);
//     if (out.is_open()) {
//         out.write(reinterpret_cast<char*>(global_model.weights), MODEL_SIZE * sizeof(float));
//         out.close();
//         std::cout << "Model global berhasil disimpan ke global_model.bin untuk inference." << std::endl;
//     } else {
//         std::cerr << "Gagal menyimpan model ke global_model.bin" << std::endl;
//     }

//     close(server_fd);
//     return 0;
// }
// // --- FUNGSI TESTING & EVALUASI REGRESI ---
// bool evaluate_model(const std::string& filepath, const float* weights) {
//     std::ifstream in(filepath, std::ios::binary);
//     if (!in.is_open()) {
//         std::cerr << "[EVALUASI ERROR] Gagal membuka file testing: " << filepath << std::endl;
//         return false;
//     }

//     int num_samples = 0;
//     in.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
//     if (num_samples <= 0) {
//         std::cerr << "[EVALUASI ERROR] Sampel testing kosong atau tidak valid." << std::endl;
//         return false;
//     }

//     std::vector<float> true_labels(num_samples);
//     std::vector<float> pred_labels(num_samples);

//     // 1. Load Data dan Lakukan Inference/Prediksi
//     for (int s = 0; s < num_samples; s++) {
//         std::vector<float> features(MODEL_SIZE);
//         in.read(reinterpret_cast<char*>(features.data()), MODEL_SIZE * sizeof(float));
//         in.read(reinterpret_cast<char*>(&true_labels[s]), sizeof(float));

//         // Prediksi linear (Dot Product)
//         float prediction = 0.0f;
//         for (int i = 0; i < MODEL_SIZE; i++) {
//             prediction += features[i] * weights[i];
//         }

//         // Batasi hasil prediksi sesuai rentang probabilitas 0 s.d 1
//         if (prediction > 1.0f) prediction = 1.0f;
//         if (prediction < 0.0f) prediction = 0.0f;

//         pred_labels[s] = prediction;
//     }
//     in.close();

//     // 2. Hitung Rata-rata Label Aktual (untuk R-squared)
//     double sum_true = 0.0;
//     #pragma omp parallel for reduction(+:sum_true)
//     for (int i = 0; i < num_samples; i++) {
//         sum_true += true_labels[i];
//     }
//     double mean_true = sum_true / num_samples;

//     // 3. Hitung Metrik Evaluasi Regresi Komparatif via OpenMP
//     double ss_residual = 0.0; // Sum of Squared Residuals
//     double ss_total = 0.0;    // Total Sum of Squares
//     double absolute_error_sum = 0.0;

//     #pragma omp parallel for reduction(+:ss_residual, ss_total, absolute_error_sum)
//     for (int i = 0; i < num_samples; i++) {
//         double diff = true_labels[i] - pred_labels[i];
//         ss_residual += diff * diff;
//         absolute_error_sum += std::abs(diff);

//         double dev = true_labels[i] - mean_true;
//         ss_total += dev * dev;
//     }

//     double mse = ss_residual / num_samples;
//     double rmse = std::sqrt(mse);
//     double mae = absolute_error_sum / num_samples;
//     double r_squared = 1.0 - (ss_residual / (ss_total + 1e-10)); // Tambah epsilon kecil anti-div-by-zero

//     // 4. Print Laporan Hasil Evaluasi
//     std::cout << "\n=========================================================" << std::endl;
//     std::cout << "        LAPORAN EVALUASI MODEL (REGRESI)                 " << std::endl;
//     std::cout << "=========================================================" << std::endl;
//     std::cout << " File Path       : " << filepath << std::endl;
//     std::cout << " Jumlah Sampel   : " << num_samples << " pelanggan" << std::endl;
//     std::cout << "---------------------------------------------------------" << std::endl;
//     std::cout << " Mean Absolute Error (MAE)      : " << mae << std::endl;
//     std::cout << " Mean Squared Error (MSE)       : " << mse << std::endl;
//     std::cout << " Root Mean Squared Error (RMSE) : " << rmse << std::endl;
//     std::cout << " R-squared (R2 Score)           : " << r_squared << " (" << r_squared * 100.0 << "%)" << std::endl;
//     std::cout << "=========================================================\n" << std::endl;

//     return true;
// }