rm -rf build

mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DUSE_CUDA=ON \
	..
make -j
