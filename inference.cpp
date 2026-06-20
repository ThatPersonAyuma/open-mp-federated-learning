#include <iostream>
#include <fstream>
#include <vector>
#include "shared.h"

void load_model(const char* filename, float* weights, int size) {
    std::ifstream in(filename, std::ios::binary);
    if (in.is_open()) {
        in.read(reinterpret_cast<char*>(weights), size * sizeof(float));
        in.close();
        std::cout << "Model berhasil dimuat dari " << filename << std::endl;
    } else {
        std::cerr << "Gagal memuat model dari " << filename << std::endl;
        exit(1);
    }
}

int main() {
    float weights[MODEL_SIZE];
    
    // Memuat model yang telah disimpan oleh master
    load_model("global_model.bin", weights, MODEL_SIZE);

    // Simulasi data pelanggan baru untuk inference
    std::vector<float> new_customer(MODEL_SIZE, 0.0f);
    
    // Fitur 0: Umur 25 tahun (dinormalisasi: 25/60)
    new_customer[0] = 25.0f / 60.0f;
    // Fitur 1: Waktu browsing 45 menit (dinormalisasi: 45/120)
    new_customer[1] = 45.0f / 120.0f;
    // Fitur 2: Jumlah transaksi 5 (dinormalisasi: 5/50)
    new_customer[2] = 5.0f / 50.0f;
    // Fitur 3: Rating rata-rata 4.5 (dinormalisasi: 4.5/5.0)
    new_customer[3] = 4.5f / 5.0f;

    // Prediksi (Inference)
    float prediction = 0.0f;
    for (int i = 0; i < MODEL_SIZE; i++) {
        prediction += new_customer[i] * weights[i];
    }

    // Batasi hasil prediksi antara 0 dan 1
    if (prediction > 1.0f) prediction = 1.0f;
    if (prediction < 0.0f) prediction = 0.0f;

    std::cout << "\n--- Hasil Inference ---" << std::endl;
    std::cout << "Profil Pelanggan Baru:" << std::endl;
    std::cout << "- Umur: 25 tahun" << std::endl;
    std::cout << "- Waktu Browsing: 45 menit" << std::endl;
    std::cout << "- Jumlah Transaksi: 5" << std::endl;
    std::cout << "- Rating Rata-rata: 4.5" << std::endl;
    std::cout << "\nProbabilitas Rekomendasi/Pembelian: " << (prediction * 100.0f) << "%" << std::endl;

    return 0;
}
