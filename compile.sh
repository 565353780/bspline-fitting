if [ "$(uname)" == "Darwin" ]; then
	PROCESSOR_NUM=$(sysctl -n hw.physicalcpu)
elif [ "$(uname)" = "Linux" ]; then
	PROCESSOR_NUM=$(cat /proc/cpuinfo | grep "processor" | wc -l)
fi

export MAX_JOBS=${PROCESSOR_NUM}

# pip uninstall bs-fit -y

# rm -rf ../bspline-fitting/build
# rm -rf ../bspline-fitting/*.egg-info
# rm ../bspline-fitting/*.so

bear -- python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
