#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "shared.h" // Pastikan file ini berisi definisi MODEL_SIZE yang sama

void run_inference(const std::string& dataset_path, const std::string& model_path) {
    // 1. MEMBUAT MODEL WEIGHTS
    std::ifstream in_model(model_path, std::ios::binary);
    if (!in_model.is_open()) {
        std::cerr << "Error: Gagal membuka file model di " << model_path << std::endl;
        return;
    }
    std::vector<float> weights(MODEL_SIZE);
    in_model.read(reinterpret_cast<char*>(weights.data()), MODEL_SIZE * sizeof(float));
    in_model.close();

    // 2. MEMBUKA DATASET BINER
    std::ifstream in_dataset(dataset_path, std::ios::binary);
    if (!in_dataset.is_open()) {
        std::cerr << "Error: Gagal membuka file dataset di " << dataset_path << std::endl;
        return;
    }

    // Baca jumlah sampel di awal file biner
    int num_samples = 0;
    in_dataset.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
    std::cout << "Berhasil memuat " << num_samples << " data dari dataset.\n";
    std::cout << "---------------------------------------------------------\n";

    std::vector<float> features(MODEL_SIZE);
    float true_label;

    // 3. LOOP UNTUK PREDIKSI TIAP ROW DATA
    for (int i = 0; i < num_samples; i++) {
        in_dataset.read(reinterpret_cast<char*>(features.data()), MODEL_SIZE * sizeof(float));
        in_dataset.read(reinterpret_cast<char*>(&true_label), sizeof(float));

        // Hitung nilai prediksi (Dot Product antara fitur dan bobot)
        float prediction = 0.0f;
        for (int j = 0; j < MODEL_SIZE; j++) {
            prediction += features[j] * weights[j];
        }

        // Batasi hasil (Clamping) antara 0.0 s.d 1.0 seperti pada generator
        if (prediction > 1.0f) prediction = 1.0f;
        if (prediction < 0.0f) prediction = 0.0f;

        // DENORMALISASI (Mengembalikan nilai 0.0 - 1.0 ke angka riil semula)
        float umur          = features[0] * (60.0f - 18.0f) + 18.0f;
        float waktu_browsing = features[1] * 120.0f;
        float transaksi     = features[2] * 50.0f;
        float rating        = features[3] * 5.0f;

        // TAMPILKAN HASIL SESUAI FORMAT YANG DIMINTA
        std::cout << "Prediksi untuk data " << (i + 1) 
                  << " dengan parameter "
                  << "umur: " << std::round(umur) << ", "
                  << "waktu: " << std::round(waktu_browsing) << " mnt, "
                  << "transaksi: " << std::round(transaksi) << ", "
                  << "rating: " << rating << ", "
                  << "hasilnya (Probabilitas Belanja): " << (prediction * 100.0f) << "%\n";
    }

    in_dataset.close();
}

int main(int argc, char* argv[]) {
    // Validasi input parameter dari terminal
    if (argc < 3) {
        std::cout << "Cara Penggunaan:\n";
        std::cout << "  " << argv[0] << " <path_file_dataset_bin> <path_file_model_bin>\n\n";
        std::cout << "Contoh:\n";
        std::cout << "  " << argv[0] << " data/dataset.bin model/global_model.bin\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    std::string model_path = argv[2];

    run_inference(dataset_path, model_path);

    return 0;
}