#ifndef SHARED_H
#define SHARED_H
#include <cstdint>
#define MODEL_SIZE 5000
// #define PORT 28080
unsigned long getPort()
{

    char* env = getenv("PORT");

    if(env)
        return atoi(env);


    return 9000;
}
#define MAGIC_NUMBER 0xABCDEF   // Token validasi worker

// Enumerasi Kontrol Operasi
enum class CommandType : int32_t {
    REQ_GENERATION_INFO = 10,  // Worker nanya generation berapa sekarang
    SEND_WORKER_UPDATE  = 20,  // Worker mau setor hasil training
    SIGNAL_SHUTDOWN     = 99   // Signal keluar dari loop
};

struct ModelPacket {
    uint32_t magic_token;
    CommandType command;       // Flag endpoint / jenis request
    int32_t current_generation;// Info generasi data saat ini
    int32_t data_size;         // Jumlah data lokal worker
    float weights[MODEL_SIZE];
};

#endif
