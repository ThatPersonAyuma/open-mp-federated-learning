#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include "shared.h"

int num_samples = 20000;

std::vector<float> local_X;
std::vector<float> local_y;

void initialize_dummy_data() {
    std::cout << "Mengalokasikan " << (num_samples * MODEL_SIZE * 4.0) / (1024*1024) << " MB RAM untuk dataset..." << std::endl;
    local_X.resize(num_samples * MODEL_SIZE);
    local_y.resize(num_samples);

    for(int i = 0; i < num_samples; i++) {
        local_y[i] = 1.0f;
        for(int j = 0; j < MODEL_SIZE; j++) {
            local_X[i * MODEL_SIZE + j] = (float)((i + j) % 100) * 0.001f;
        }
    }
    std::cout << "Inisialisasi dataset selesai.\n" << std::endl;
}

void train_local_model(ModelPacket &packet, int epochs, float lr) {
    std::cout << "Mulai Heavy Local Training (OpenMP: " << omp_get_max_threads() << " threads)..." << std::endl;
    double start_time = omp_get_wtime();

    int batch_size = 256; 

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int b = 0; b < num_samples; b += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - b);
            std::vector<float> batch_gradients(MODEL_SIZE, 0.0f);

            #pragma omp parallel
            {
                std::vector<float> local_gradients(MODEL_SIZE, 0.0f);

                #pragma omp for schedule(static)
                for (int i = b; i < b + current_batch_size; i++) {
                    float prediction = 0.0f;
                    int base_idx = i * MODEL_SIZE;

                    for (int j = 0; j < MODEL_SIZE; j++) {
                        prediction += local_X[base_idx + j] * packet.weights[j];
                    }

                    float error = prediction - local_y[i];

                    // 1. Error Clipping (Mencegah prediksi liar)
                    if (error > 5.0f) error = 5.0f;
                    if (error < -5.0f) error = -5.0f;

                    for (int j = 0; j < MODEL_SIZE; j++) {
                        local_gradients[j] += error * local_X[base_idx + j];
                    }
                }

                #pragma omp critical
                {
                    for (int j = 0; j < MODEL_SIZE; j++) {
                        batch_gradients[j] += local_gradients[j];
                    }
                }
            }

            // Update bobot + 2. Gradient Clipping
            for (int j = 0; j < MODEL_SIZE; j++) {
                float grad = batch_gradients[j] / current_batch_size;
                
                // Batasi gradien maksimal demi kestabilan matematika
                if (grad > 1.0f) grad = 1.0f;
                if (grad < -1.0f) grad = -1.0f;

                packet.weights[j] -= lr * grad;
            }
        }

        // Cek deteksi NaN
        if (std::isnan(packet.weights[0])) {
            std::cerr << "Kritis: Bobot mendeteksi NaN di Epoch " << epoch + 1 << "!" << std::endl;
            exit(EXIT_FAILURE);
        }

        if ((epoch + 1) % 10 == 0) {
            std::cout << "  -> Progress: Epoch " << epoch + 1 << "/" << epochs << " selesai." << std::endl;
        }
    }

    double end_time = omp_get_wtime();
    packet.data_size = num_samples;
    std::cout << "\n[!] Local Training Selesai!" << std::endl;
    std::cout << "[!] Waktu Komputasi Worker: " << (end_time - start_time) << " detik." << std::endl;
}

int main() {
    initialize_dummy_data();

    int sock = 0;
    struct sockaddr_in serv_addr;
    ModelPacket packet;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Ganti dengan IP Master Node LAN
    if (inet_pton(AF_INET, "192.168.34.253", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return -1;
    }

    std::cout << "Menghubungkan ke Master..." << std::endl;
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection Failed" << std::endl;
        return -1;
    }

    // Terima & Validasi
    int total_received = 0;
    char* recv_ptr = (char*)&packet;
    while (total_received < sizeof(ModelPacket)) {
        int bytes = recv(sock, recv_ptr + total_received, sizeof(ModelPacket) - total_received, 0);
        if (bytes <= 0) {
            std::cerr << "Gagal menerima data dari Master" << std::endl;
            break;
        }
        total_received += bytes;
    }

    if (packet.magic_token == MAGIC_NUMBER) {
        std::cout << "Model Global (Valid) diterima dari Master." << std::endl;
    }

    // Eksekusi Training (50 Epochs)
    train_local_model(packet, 50, 0.001f);

    // Kirim Balik
    int total_sent = 0;
    char* send_ptr = (char*)&packet;
    while (total_sent < sizeof(ModelPacket)) {
        int bytes = send(sock, send_ptr + total_sent, sizeof(ModelPacket) - total_sent, 0);
        if (bytes <= 0) {
            std::cerr << "Gagal mengirim data ke Master" << std::endl;
            break;
        }
        total_sent += bytes;
    }
    std::cout << "Update bobot berhasil dikirim ke Master." << std::endl;

    shutdown(sock, SHUT_WR);
    close(sock);
    return 0;
}