
# Neural Network in C

This is assignment 1

## Compile & Run

* sh is for ubuntu command line (Bash)
```sh
# Compile
gcc -c src/main.c -o build/main.o
gcc -c src/func.c -o build/func.o
gcc build/main.o build/func.o -o bin/nncc -lm
# Run
./bin/nncc
```
