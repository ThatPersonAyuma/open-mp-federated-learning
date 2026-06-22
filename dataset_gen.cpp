#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include "shared.h"
#include <string>

enum class Scenario {
    AWAL_TREND,    // Niche market: Kaum muda, browsing singkat, transaksi sedikit, rating tinggi
    SAAT_INI       // Mass market: Semua umur, browsing lama, transaksi banyak, rating variatif
};

struct Hyperparameters {
    Scenario tipe_skenario = Scenario::SAAT_INI; // Pilih skenario di sini
    float base_bias = 0.01f;
    float noise_level = 0.02f;
    int num_samples = 1000;
};

// 1. MEMBUAT TREN BOBOT GLOBAL BERDASARKAN SKENARIO
void generate_global_model(const char* filename, const Hyperparameters& hp) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Gagal membuat file model global!" << std::endl;
        return;
    }

    std::vector<float> weights(MODEL_SIZE);
    
    for (int i = 0; i < MODEL_SIZE; i++) {
        float progress = static_cast<float>(i) / MODEL_SIZE;
        
        if (hp.tipe_skenario == Scenario::AWAL_TREND) {
            // Skenario Awal: Hanya fitur-fitur spesifik (seperti teknologi/spesifik barang) yang berpengaruh besar
            weights[i] = hp.base_bias + (0.3f * std::sin(progress * 3.14f));
        } else {
            // Skenario Saat Ini: Dampak fitur lebih merata dan masif di seluruh dimensi karena barang beragam
            weights[i] = hp.base_bias + (0.5f * progress) + (0.1f * std::cos(progress * 6.28f));
        }
    }

    // Berikan bobot manual yang kuat pada 4 fitur utama agar hasil inference mencerminkan skenario
    if (hp.tipe_skenario == Scenario::AWAL_TREND) {
        weights[0] = 0.8f;  // Umur muda sangat berpengaruh positif
        weights[1] = 0.4f;  // Browsing singkat sudah cukup membeli
        weights[2] = 0.2f;  // Transaksi sedikit
        weights[3] = 0.9f;  // Rating harus tinggi (hanya beli barang yang pasti bagus)
    } else {
        weights[0] = 0.3f;  // Umur tidak terlalu mendominasi lagi (merata)
        weights[1] = 0.8f;  // Browsing lama berkorelasi tinggi dengan pembelian
        weights[2] = 0.9f;  // Jumlah transaksi tinggi adalah pendorong utama
        weights[3] = 0.5f;  // Rating 3-5 pun tetap dibeli
    }

    out.write(reinterpret_cast<const char*>(weights.data()), MODEL_SIZE * sizeof(float));
    out.close();
    std::cout << "Sukses: Model Global untuk Skenario " 
              << (hp.tipe_skenario == Scenario::AWAL_TREND ? "Awal Trend" : "Saat Ini") 
              << " berhasil dibuat.\n";
}

// 2. GENERATOR DATASET BERDASARKAN DISTRIBUSI SKENARIO
void generate_dataset(const char* filename, const Hyperparameters& hp, const char* model_filename) {
    std::ifstream in_model(model_filename, std::ios::binary);
    std::vector<float> true_weights(MODEL_SIZE);
    if (in_model.is_open()) {
        in_model.read(reinterpret_cast<char*>(true_weights.data()), MODEL_SIZE * sizeof(float));
        in_model.close();
    } else {
        std::cerr << "Gagal membaca model acuan!" << std::endl;
        return;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_generic(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, hp.noise_level);

    out.write(reinterpret_cast<const char*>(&hp.num_samples), sizeof(int));

    for (int s = 0; s < hp.num_samples; s++) {
        std::vector<float> features(MODEL_SIZE);
        float raw_umur, raw_browsing, raw_transaksi, raw_rating;

        if (hp.tipe_skenario == Scenario::AWAL_TREND) {
            std::uniform_real_distribution<float> dist_umur(18.0f, 30.0f);
            std::uniform_real_distribution<float> dist_browsing(15.0f, 45.0f);
            std::uniform_real_distribution<float> dist_transaksi(1.0f, 5.0f);
            std::uniform_real_distribution<float> dist_rating(4.0f, 5.0f);

            raw_umur = dist_umur(gen);
            raw_browsing = dist_browsing(gen);
            raw_transaksi = dist_transaksi(gen);
            raw_rating = dist_rating(gen);
        } 
        else {
            std::uniform_real_distribution<float> p_dist(0.0f, 100.0f);
            float p = p_dist(gen);
            if (p <= 50.0f)       raw_umur = std::uniform_real_distribution<float>(18.0f, 30.0f)(gen);
            else if (p <= 90.0f)  raw_umur = std::uniform_real_distribution<float>(30.0f, 45.0f)(gen);
            else                  raw_umur = std::uniform_real_distribution<float>(46.0f, 60.0f)(gen);

            std::uniform_real_distribution<float> dist_browsing(100.0f, 120.0f);
            std::uniform_real_distribution<float> dist_transaksi(20.0f, 50.0f);
            std::uniform_real_distribution<float> dist_rating(3.0f, 5.0f);

            raw_browsing = dist_browsing(gen);
            raw_transaksi = dist_transaksi(gen);
            raw_rating = dist_rating(gen);
        }

        // MIN-MAX NORMALIZATION
        features[0] = (raw_umur - 18.0f) / (60.0f - 18.0f);
        features[1] = raw_browsing / 120.0f;
        features[2] = raw_transaksi / 50.0f;
        features[3] = raw_rating / 5.0f;

        // Isi sisa fitur laten (4 s.d 4999) - Menggunakan distribusi acak murni per sampel
        for (int i = 4; i < MODEL_SIZE; i++) {
            features[i] = dist_generic(gen);
        }

        // --- FEATURE ENGINEERING: IMPLEMENTASI DISTRIBUSI RASIO 70% : 30% ---
        
        // 1. Ambil kontribusi murni dari 4 fitur utama (Komponen Utama)
        float main_features_sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            main_features_sum += features[i] * true_weights[i];
        }

        // 2. Ambil rata-rata kontribusi dari 4996 fitur laten (Komponen Laten)
        float latent_features_sum = 0.0f;
        for (int i = 4; i < MODEL_SIZE; i++) {
            latent_features_sum += features[i] * true_weights[i];
        }
        float latent_features_avg = latent_features_sum / (MODEL_SIZE - 4);

        // 3. Gabungkan dengan pembobotan proporsional (70% Utama, 30% Laten)
        // Catatan: latent_features_avg dikalikan kembali dengan rata-rata base weight agar skalanya seimbang
        float base_scale = 0.01f; // Menyesuaikan bobot inisial global model (0.01f)
        float label = (0.70f * main_features_sum) + (0.30f * latent_features_avg * base_scale);
        
        // 4. Tambahkan noise level target
        label += noise_dist(gen);

        // Batasi rentang probabilitas regresi (0.0 s.d 1.0)
        if (label > 1.0f) label = 1.0f;
        if (label < 0.0f) label = 0.0f;

        // Tulis data biner
        out.write(reinterpret_cast<const char*>(features.data()), MODEL_SIZE * sizeof(float));
        out.write(reinterpret_cast<const char*>(&label), sizeof(float));
    }

    out.close();
    std::cout << "Sukses: Dataset simulasi dengan kontrol kontribusi 70:30 berhasil dibuat.\n";
}
// void generate_dataset_federated(const char* base_filename, const char* test_base_path, const Hyperparameters& hp, const char* model_filename, int num_workers) {
//     if (num_workers <= 0) {
//         std::cerr << "Jumlah worker harus minimal 1!" << std::endl;
//         return;
//     }

//     // 1. MEMBACA MODEL ACUAN
//     std::ifstream in_model(model_filename, std::ios::binary);
//     std::vector<float> true_weights(MODEL_SIZE);
//     if (in_model.is_open()) {
//         in_model.read(reinterpret_cast<char*>(true_weights.data()), MODEL_SIZE * sizeof(float));
//         in_model.close();
//     } else {
//         std::cerr << "Gagal membaca model acuan!" << std::endl;
//         return;
//     }

//     // 2. HITUNG ALOKASI DATA (20% TEST SET, SISA 80% UNTUK WORKER)
//     int num_test_samples = static_cast<int>(hp.num_samples * 0.20);
//     int num_worker_samples = hp.num_samples - num_test_samples;

//     // 3. MENYIAPKAN FILE DATA TEST
//     std::string test_filename = std::string(test_base_path) + ".bin";
//     std::ofstream test_file(test_filename, std::ios::binary);
//     if (!test_file.is_open()) {
//         std::cerr << "Gagal membuat file test di " << test_filename << std::endl;
//         return;
//     }
//     int placeholder = 0;
//     test_file.write(reinterpret_cast<const char*>(&placeholder), sizeof(int));
//     int test_counter = 0;

//     // 4. MENYIAPKAN FILE STREAM & COUNTER UNTUK TIAP WORKER
//     std::vector<std::ofstream> out_files(num_workers);
//     std::vector<int> worker_sample_counts(num_workers, 0);

//     for (int i = 0; i < num_workers; i++) {
//         std::string filename = std::string(base_filename) + "_worker_" + std::to_string(i) + ".bin";
//         out_files[i].open(filename, std::ios::binary);
//         if (!out_files[i].is_open()) {
//             std::cerr << "Gagal membuat file untuk worker " << i << std::endl;
//             return;
//         }
//         out_files[i].write(reinterpret_cast<const char*>(&placeholder), sizeof(int));
//     }

//     // 5. MENYIAPKAN GENERATOR ACAK
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist_generic(0.0f, 1.0f);
//     std::uniform_real_distribution<float> dist_routing(0.0f, 1.0f); // Untuk routing test vs worker
//     std::normal_distribution<float> noise_dist(0.0f, hp.noise_level);
//     std::uniform_int_distribution<int> worker_dist(0, num_workers - 1);

//     // 6. LOOP GENERASI DATA (MENGGUNAKAN TOTAL AWAL SAMPLES)
//     for (int s = 0; s < hp.num_samples; s++) {
//         std::vector<float> features(MODEL_SIZE);
//         float raw_umur, raw_browsing, raw_transaksi, raw_rating;

//         if (hp.tipe_skenario == Scenario::AWAL_TREND) {
//             std::uniform_real_distribution<float> dist_umur(18.0f, 30.0f);
//             std::uniform_real_distribution<float> dist_browsing(15.0f, 45.0f);
//             std::uniform_real_distribution<float> dist_transaksi(1.0f, 5.0f);
//             std::uniform_real_distribution<float> dist_rating(4.0f, 5.0f);

//             raw_umur = dist_umur(gen);
//             raw_browsing = dist_browsing(gen);
//             raw_transaksi = dist_transaksi(gen);
//             raw_rating = dist_rating(gen);
//         } 
//         else {
//             std::uniform_real_distribution<float> p_dist(0.0f, 100.0f);
//             float p = p_dist(gen);
//             if (p <= 50.0f)       raw_umur = std::uniform_real_distribution<float>(18.0f, 30.0f)(gen);
//             else if (p <= 90.0f)  raw_umur = std::uniform_real_distribution<float>(30.0f, 45.0f)(gen);
//             else                  raw_umur = std::uniform_real_distribution<float>(46.0f, 60.0f)(gen);

//             std::uniform_real_distribution<float> dist_browsing(100.0f, 120.0f);
//             std::uniform_real_distribution<float> dist_transaksi(20.0f, 50.0f);
//             std::uniform_real_distribution<float> dist_rating(3.0f, 5.0f);

//             raw_browsing = dist_browsing(gen);
//             raw_transaksi = dist_transaksi(gen);
//             raw_rating = dist_rating(gen);
//         }

//         // MIN-MAX NORMALIZATION
//         features[0] = (raw_umur - 18.0f) / (60.0f - 18.0f);
//         features[1] = raw_browsing / 120.0f;
//         features[2] = raw_transaksi / 50.0f;
//         features[3] = raw_rating / 5.0f;

//         // Isi sisa fitur laten (4 s.d 4999) - Menggunakan distribusi acak murni per sampel
//         for (int i = 4; i < MODEL_SIZE; i++) {
//             features[i] = dist_generic(gen);
//         }

//         // --- FEATURE ENGINEERING: IMPLEMENTASI DISTRIBUSI RASIO 70% : 30% ---
        
//         // 1. Ambil kontribusi murni dari 4 fitur utama (Komponen Utama)
//         float main_features_sum = 0.0f;
//         for (int i = 0; i < 4; i++) {
//             main_features_sum += features[i] * true_weights[i];
//         }

//         // 2. Ambil rata-rata kontribusi dari 4996 fitur laten (Komponen Laten)
//         float latent_features_sum = 0.0f;
//         for (int i = 4; i < MODEL_SIZE; i++) {
//             latent_features_sum += features[i] * true_weights[i];
//         }
//         float latent_features_avg = latent_features_sum / (MODEL_SIZE - 4);

//         // 3. Gabungkan dengan pembobotan proporsional (70% Utama, 30% Laten)
//         // Catatan: latent_features_avg dikalikan kembali dengan rata-rata base weight agar skalanya seimbang
//         float base_scale = 0.01f; // Menyesuaikan bobot inisial global model (0.01f)
//         float label = (0.70f * main_features_sum) + (0.30f * latent_features_avg * base_scale);
        
//         // 4. Tambahkan noise level target
//         label += noise_dist(gen);

//         // Batasi rentang probabilitas regresi (0.0 s.d 1.0)
//         if (label > 1.0f) label = 1.0f;
//         if (label < 0.0f) label = 0.0f;

//         // --- 7. STRATEGI PEMISAHAN DATA SECARA ACAK ---
//         bool send_to_test = false;
//         int remaining_total_samples = hp.num_samples - s;
//         int needed_test_samples = num_test_samples - test_counter;

//         if (needed_test_samples > 0) {
//             if (remaining_total_samples <= needed_test_samples) {
//                 send_to_test = true; // Sisa data wajib dialokasikan ke test set agar kuota pas
//             } else {
//                 // Peluang dinamis disesuaikan secara real-time agar pembagian acak merata
//                 float prob = static_cast<float>(needed_test_samples) / remaining_total_samples;
//                 if (dist_routing(gen) < prob) {
//                     send_to_test = true;
//                 }
//             }
//         }

//         // Tulis data ke file tujuan yang tepat
//         if (send_to_test) {
//             test_file.write(reinterpret_cast<const char*>(features.data()), MODEL_SIZE * sizeof(float));
//             test_file.write(reinterpret_cast<const char*>(&label), sizeof(float));
//             test_counter++;
//         } else {
//             // Masuk ke pool 80% data worker, lalu diundi lagi dapet worker mana secara acak
//             int chosen_worker = worker_dist(gen);
//             out_files[chosen_worker].write(reinterpret_cast<const char*>(features.data()), MODEL_SIZE * sizeof(float));
//             out_files[chosen_worker].write(reinterpret_cast<const char*>(&label), sizeof(float));
//             worker_sample_counts[chosen_worker]++;
//         }
//     }

//     // 8. KOREKSI DATA HEADER UNTUK FILE TEST
//     test_file.seekp(0, std::ios::beg);
//     test_file.write(reinterpret_cast<const char*>(&test_counter), sizeof(int));
//     test_file.close();

//     // 9. KOREKSI DATA HEADER UNTUK SETIAP FILE WORKER
//     std::cout << "\n================= DISTRIBUSI DATASET =================" << std::endl;
//     std::cout << " [TEST SET]  -> " << test_counter << " sampel (" << test_filename << ")" << std::endl;
//     std::cout << " -----------------------------------------------------" << std::endl;
//     for (int i = 0; i < num_workers; i++) {
//         out_files[i].seekp(0, std::ios::beg);
//         out_files[i].write(reinterpret_cast<const char*>(&worker_sample_counts[i]), sizeof(int));
//         out_files[i].close();
//         std::cout << " [WORKER " << i << "] -> " << worker_sample_counts[i] << " sampel." << std::endl;
//     }
//     std::cout << "======================================================" << std::endl;
// }

void generate_dataset_federated(const char* base_filename, const char* test_base_path, const Hyperparameters& hp, const char* model_filename, int num_workers, int generation) {
    if (num_workers <= 0) {
        std::cerr << "Jumlah worker harus minimal 1!" << std::endl;
        return;
    }

    // 1. MEMBACA MODEL ACUAN
    std::ifstream in_model(model_filename, std::ios::binary);
    std::vector<float> true_weights(MODEL_SIZE);
    if (in_model.is_open()) {
        in_model.read(reinterpret_cast<char*>(true_weights.data()), MODEL_SIZE * sizeof(float));
        in_model.close();
    } else {
        std::cerr << "Gagal membaca model acuan!" << std::endl;
        return;
    }

    // 2. HITUNG ALOKASI DATA (20% TEST SET, SISA 80% UNTUK WORKER)
    int num_test_samples = static_cast<int>(hp.num_samples * 0.20);
    int num_worker_samples = hp.num_samples - num_test_samples;

    // 3. MENYIAPKAN FILE DATA TEST
    std::string test_filename = std::string(test_base_path) + ".bin";
    std::ofstream test_file(test_filename, std::ios::binary);
    if (!test_file.is_open()) {
        std::cerr << "Gagal membuat file test di " << test_filename << std::endl;
        return;
    }
    int placeholder = 0;
    test_file.write(reinterpret_cast<const char*>(&placeholder), sizeof(int));
    int test_counter = 0;

    // 4. MENYIAPKAN FILE STREAM & COUNTER UNTUK TIAP WORKER
    std::vector<std::ofstream> out_files(num_workers);
    std::vector<int> worker_sample_counts(num_workers, 0);

    for (int i = 0; i < num_workers; i++) {
        std::string filename = std::string(base_filename) + "_worker_" + std::to_string(i) + ".bin";
        out_files[i].open(filename, std::ios::binary);
        if (!out_files[i].is_open()) {
            std::cerr << "Gagal membuat file untuk worker " << i << std::endl;
            return;
        }
        out_files[i].write(reinterpret_cast<const char*>(&placeholder), sizeof(int));
    }

    // 5. MENYIAPKAN GENERATOR ACAK
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_generic(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_routing(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, hp.noise_level);
    std::uniform_int_distribution<int> worker_dist(0, num_workers - 1);

    // --- STRATEGI TREND: MENGHITUNG KOEFISIEN PERGESERAN GRADUAL (DATA DRIFT) ---
    // Mengubah pergeseran biner (Past vs Now) menjadi interpolasi linear berbasis generasi (0 s.d 9)
    float t = static_cast<float>(generation) / 9.0f; // Nilai berkisar antara 0.0 hingga 1.0

    // Batas dinamis untuk Waktu Browsing (Beralih dari maks 45m ke maks 120m secara bertahap)
    float max_browsing = 45.0f + t * (120.0f - 45.0f);
    // Batas dinamis untuk Jumlah Transaksi (Beralih dari maks 5 ke maks 50 secara bertahap)
    float max_transaksi = 5.0f + t * (50.0f - 5.0f);

    // 6. LOOP GENERASI DATA
    for (int s = 0; s < hp.num_samples; s++) {
        std::vector<float> features(MODEL_SIZE);
        float raw_umur, raw_browsing, raw_transaksi, raw_rating;

        // Distribusi Umur Gradual: Semakin tinggi generasi, porsi lintas usia (30-60 tahun) semakin besar
        float p_umur = std::uniform_real_distribution<float>(0.0f, 100.0f)(gen);
        float batas_muda = 100.0f - (t * 50.0f); // Gen 0 = 100% muda, Gen 9 = 50% muda
        float batas_paruh = batas_muda + (t * 40.0f);

        if (p_umur <= batas_muda) {
            raw_umur = std::uniform_real_distribution<float>(18.0f, 30.0f)(gen);
        } else if (p_umur <= batas_paruh) {
            raw_umur = std::uniform_real_distribution<float>(30.0f, 45.0f)(gen);
        } else {
            raw_umur = std::uniform_real_distribution<float>(46.0f, 60.0f)(gen);
        }

        // Distribusi Browsing dan Transaksi Gradual mengikuti pergeseran tren waktu
        raw_browsing = std::uniform_real_distribution<float>(15.0f, max_browsing)(gen);
        raw_transaksi = std::uniform_real_distribution<float>(1.0f, max_transaksi)(gen);
        
        // Rating bergeser perlahan dari rentang (4-5) ke rentang variatif (3-5)
        float min_rating = 4.0f - t * 1.0f; 
        raw_rating = std::uniform_real_distribution<float>(min_rating, 5.0f)(gen);

        // MIN-MAX NORMALIZATION (0.0 s.d 1.0)
        features[0] = (raw_umur - 18.0f) / (60.0f - 18.0f);
        features[1] = raw_browsing / 120.0f;
        features[2] = raw_transaksi / 50.0f;
        features[3] = raw_rating / 5.0f;

        // Isi sisa fitur laten
        for (int i = 4; i < MODEL_SIZE; i++) {
            features[i] = dist_generic(gen);
        }

        // --- FEATURE ENGINEERING: TIME-VARYING WEIGHTS & RASIO CONTROLLER ---
        
        // Memodifikasi sensitivitas target secara dinamis seiring bertambahnya generasi (Trend Kausalitas)
        float trend_factor_umur = 1.0f - (t * 0.30f);     // Pengaruh umur melemah 30% di akhir generasi
        float trend_factor_browsing = 1.0f + (t * 0.40f); // Pengaruh waktu browsing menguat 40%

        // 1. Ambil kontribusi dinamis dari 4 fitur utama 
        float main_features_sum = 0.0f;
        main_features_sum += features[0] * (true_weights[0] * trend_factor_umur);
        main_features_sum += features[1] * (true_weights[1] * trend_factor_browsing);
        main_features_sum += features[2] * true_weights[2];
        main_features_sum += features[3] * true_weights[3];

        // 2. Ambil rata-rata kontribusi dari 4996 fitur laten
        float latent_features_sum = 0.0f;
        for (int i = 4; i < MODEL_SIZE; i++) {
            latent_features_sum += features[i] * true_weights[i];
        }
        float latent_features_avg = latent_features_sum / (MODEL_SIZE - 4);

        // 3. Gabungkan dengan pembobotan proporsional (70% Utama, 30% Laten) + Efek Musiman (Sinusoidal)
        float base_scale = 0.01f;
        float seasonality = 0.05f * std::sin(t * 3.14159f * 2.0f); // Siklus musiman naik turun sepanjang generasi
        
        float label = (0.70f * main_features_sum) + (0.30f * latent_features_avg * base_scale) + seasonality;
        
        // 4. Tambahkan noise level target
        label += noise_dist(gen);

        // Batasi rentang probabilitas regresi (0.0 s.d 1.0)
        if (label > 1.0f) label = 1.0f;
        if (label < 0.0f) label = 0.0f;

        // --- 7. STRATEGI PEMISAHAN DATA SECARA ACAK ---
        bool send_to_test = false;
        int remaining_total_samples = hp.num_samples - s;
        int needed_test_samples = num_test_samples - test_counter;

        if (needed_test_samples > 0) {
            if (remaining_total_samples <= needed_test_samples) {
                send_to_test = true;
            } else {
                float prob = static_cast<float>(needed_test_samples) / remaining_total_samples;
                if (dist_routing(gen) < prob) {
                    send_to_test = true;
                }
            }
        }

        // Tulis data ke file tujuan yang tepat
        if (send_to_test) {
            test_file.write(reinterpret_cast<const char*>(features.data()), MODEL_SIZE * sizeof(float));
            test_file.write(reinterpret_cast<const char*>(&label), sizeof(float));
            test_counter++;
        } else {
            int chosen_worker = worker_dist(gen);
            out_files[chosen_worker].write(reinterpret_cast<const char*>(features.data()), MODEL_SIZE * sizeof(float));
            out_files[chosen_worker].write(reinterpret_cast<const char*>(&label), sizeof(float));
            worker_sample_counts[chosen_worker]++;
        }
    }

    // 8. KOREKSI DATA HEADER UNTUK FILE TEST
    test_file.seekp(0, std::ios::beg);
    test_file.write(reinterpret_cast<const char*>(&test_counter), sizeof(int));
    test_file.close();

    // 9. KOREKSI DATA HEADER UNTUK SETIAP FILE WORKER
    std::cout << "\n================= DISTRIBUSI DATASET (GEN " << generation << ") =================" << std::endl;
    std::cout << " [TEST SET]  -> " << test_counter << " sampel (" << test_filename << ")" << std::endl;
    std::cout << " -----------------------------------------------------" << std::endl;
    for (int i = 0; i < num_workers; i++) {
        out_files[i].seekp(0, std::ios::beg);
        out_files[i].write(reinterpret_cast<const char*>(&worker_sample_counts[i]), sizeof(int));
        out_files[i].close();
        std::cout << " [WORKER " << i << "] -> " << worker_sample_counts[i] << " sampel." << std::endl;
    }
    std::cout << "======================================================" << std::endl;
}

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

int main(int argc, char* argv[]) {
    Hyperparameters hp;
    hp.num_samples = 1000; // Total data keseluruhan yang akan dipecah
    hp.noise_level = 0.02f; // Pastikan noise level terdefinisi

    int jumlah_worker = 2; // Tentukan ingin dibagi ke berapa node/worker
    const char* model_file = "model/global_model.bin";
    
    // Default nilai generasi jika tidak diinput oleh user
    int generasi_aktif = 0; 

    // Cek jika argumen pertama adalah "now"
    if (argc >= 2 && std::string(argv[1]) == "now") {
        hp.tipe_skenario = Scenario::SAAT_INI; 
        const char* dataset_file = "data/dataset_now";
        const char* dataset_test = "data/dataset_test_now"; 
        float weights[MODEL_SIZE];
        
        // FITUR DINAMIS: Mengambil angka generasi dari argumen kedua jika ada (misal: ./generator now 5)
        if (argc >= 3) {
            generasi_aktif = std::stoi(argv[2]);
        } else {
            generasi_aktif = 9; // Default puncak tren "now" di akhir timeline
        }
    
        load_model(model_file, weights, MODEL_SIZE);
        
        // Panggil dengan menyertakan parameter generasi_aktif
        generate_dataset_federated(dataset_file, dataset_test, hp, model_file, jumlah_worker, generasi_aktif);
    } 
    else {
        // Skenario "past" atau default awal trend
        hp.tipe_skenario = Scenario::AWAL_TREND; 
        const char* dataset_file = "data/dataset_past";
        const char* dataset_test = "data/dataset_test_past"; 
        
        // FITUR DINAMIS: Mengambil angka generasi untuk fase awal jika ada (misal: ./generator past 2)
        if (argc >= 3) {
            generasi_aktif = std::stoi(argv[2]);
        } else {
            generasi_aktif = 0; // Default awal mula trend data
        }

        generate_global_model(model_file, hp);
        
        // Panggil dengan menyertakan parameter generasi_aktif
        generate_dataset_federated(dataset_file, dataset_test, hp, model_file, jumlah_worker, generasi_aktif);
    }

    return 0;
}

// int main(int argc, char* argv[]) {
//     Hyperparameters hp;
//     hp.tipe_skenario = Scenario::SAAT_INI; 
//     hp.num_samples = 1000; // Total data keseluruhan yang akan dipecah

//     int jumlah_worker = 2; // Tentukan ingin dibagi ke berapa node/worker

//     const char* model_file = "model/global_model.bin";

//     // Base name tanpa ekstensi .bin
//     if (argc >= 2 && std::string(argv[1]) == "now") {
//         const char* dataset_file = "data/dataset_now";
//         const char* dataset_test = "data/dataset_test_now"; 
//         float weights[MODEL_SIZE];
    
//         // Perbaikan 4: Samakan path model dengan yang disimpan oleh master
//         load_model(model_file, weights, MODEL_SIZE);
        
//         // Perbaikan 2: model_file sekarang sudah dapat diakses di sini
//         generate_dataset_federated(dataset_file, dataset_test, hp, model_file, jumlah_worker);
//     } else {
//         const char* dataset_file = "data/dataset_past";
//         const char* dataset_test = "data/dataset_test_past"; 
//         generate_global_model(model_file, hp);
//         generate_dataset_federated(dataset_file, dataset_test, hp, model_file, jumlah_worker);
//     }

//     return 0;
// }

// int main(int argc, char* argv[]) {
//     Hyperparameters hp;
    
//     // Silakan ganti menjadi Scenario::AWAL_TREND jika ingin mensimulasikan masa lalu
//     hp.tipe_skenario = Scenario::SAAT_INI; 
//     hp.num_samples = 800;

//     // Definisikan path model di luar if-else agar bisa diakses di kedua blok
//     const char* model_file = "model/global_model.bin";

//     // Perbaikan 1 & 3: Gunakan std::string untuk perbandingan dan perbaiki batas argc
//     if (argc >= 2 && std::string(argv[1]) == "now") {
//         const char* dataset_file = "data/dataset_now.bin";
//         float weights[MODEL_SIZE];
    
//         // Perbaikan 4: Samakan path model dengan yang disimpan oleh master
//         load_model(model_file, weights, MODEL_SIZE);
        
//         // Perbaikan 2: model_file sekarang sudah dapat diakses di sini
//         generate_dataset(dataset_file, hp, model_file);
//     } else {
//         const char* dataset_file = "data/dataset.bin";
//         generate_global_model(model_file, hp);
//         generate_dataset(dataset_file, hp, model_file);
//     }

//     return 0;
// }