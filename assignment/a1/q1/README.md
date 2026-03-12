# Q1 Commands

## Compile

```bash
g++ -O2 q1a.cpp -o q1a_serial
g++ -O2 -fopenmp q1a.cpp -o q1a_parallel

g++ -O2 q1b.cpp -o q1b_serial
g++ -O2 -fopenmp q1b.cpp -o q1b_parallel
```

## Run Q1(a) Riemann

```bash
./q1a_serial

OMP_NUM_THREADS=2 ./q1a_parallel
OMP_NUM_THREADS=4 ./q1a_parallel
OMP_NUM_THREADS=8 ./q1a_parallel
```

## Run Q1(b) Monte Carlo (N = 100, 1000, 10000, 100000, 1000000)

```bash
./q1b_serial 100
./q1b_serial 1000
./q1b_serial 10000
./q1b_serial 100000
./q1b_serial 1000000

OMP_NUM_THREADS=2 ./q1b_parallel 100
OMP_NUM_THREADS=2 ./q1b_parallel 1000
OMP_NUM_THREADS=2 ./q1b_parallel 10000
OMP_NUM_THREADS=2 ./q1b_parallel 100000
OMP_NUM_THREADS=2 ./q1b_parallel 1000000

OMP_NUM_THREADS=4 ./q1b_parallel 100
OMP_NUM_THREADS=4 ./q1b_parallel 1000
OMP_NUM_THREADS=4 ./q1b_parallel 10000
OMP_NUM_THREADS=4 ./q1b_parallel 100000
OMP_NUM_THREADS=4 ./q1b_parallel 1000000

OMP_NUM_THREADS=8 ./q1b_parallel 100
OMP_NUM_THREADS=8 ./q1b_parallel 1000
OMP_NUM_THREADS=8 ./q1b_parallel 10000
OMP_NUM_THREADS=8 ./q1b_parallel 100000
OMP_NUM_THREADS=8 ./q1b_parallel 1000000
```

## Plot Commands

```bash
uv run q1aplot.py
uv run q1bplot.py
```
