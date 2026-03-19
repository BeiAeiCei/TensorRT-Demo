cd Myplugin/
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
make -j$(nproc)


cd ../
