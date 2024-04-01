cd ..
git clone https://github.com/565353780/geom-dl.git

cd geom-dl
./setup.sh

if [ "$(uname)" == "Darwin" ]; then
	brew install bear
	pip install open3d==0.15.1
elif [ "$(uname)" = "Linux" ]; then
	sudo apt install bear -y
	pip install -U open3d
fi

pip install -U ninja

pip install -U torch torchvision torchaudio

cd ../bspline-fitting
./compile.sh

pip install -U tqdm tensorboard matplotlib gradio plotly
