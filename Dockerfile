FROM ubuntu:24.04


RUN apt update && apt install -y \
    clang \
    cmake \
    make \
    libomp-dev


WORKDIR /app


COPY . .


RUN cmake -B build \
    && cmake --build build


CMD ["./build/masterNode"]