#ifndef SHARED_H
#define SHARED_H

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

typedef struct {
    int magic_token;            // Penanda kalo paket dari worker asli
    float weights[MODEL_SIZE];
    int data_size; 
} ModelPacket;

#endif
