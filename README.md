# gpu_extractor

## How to run
```
~$ mkdir build
~$ cd build
~$ cmake ../
~$ make
~$ ./main/simple
```

## Quick Start
```
git clone <repository-url>
cd gpu_extraction

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure # for all unittests

ctest -R test_gpu_batch_then_direct --output-on-failure # for specific unittest called `test_gpu_batch_then_direct`

cd ..
./run.sh
```