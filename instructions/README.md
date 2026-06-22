# For Docker Usage

1. Dataset
You need to create dataset. It should be portable as it didn't use any OS tied API
```bash
g++ dataset_gen.cpp -o dataset_gen
```
For past dataset use (it is also create the base model):
```bash
./dataset_gen
```
Then, for now dataset:
```bash
./dataset_gen now
```

2. Running Docker
```bash
docker compose build
docker compose run
```
Or
```bash
docker compose up -build
```

Also as it's print the log, I suggest using docker desktop for more prettier log.