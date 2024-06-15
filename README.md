# Dorylus Elastic Agent Engine
This appliaction is running in Hajime edge nodes or other x86 pc with GPU, providing automatic speech recognition (ASR), connections to large language model,
retrieval-augmented generation and text to speech etc. The node applications connect to the central schedule hub server, and communicate with other nodes.

# Prerequisites
- Ubuntu 22.04.4 LTS
- Python 3.10
- Cuda 12.1

# Installation
```
# get codes from git
git clone <git-repo-url>
cd <path-to-node-app>
mkdir -p cache/speech/record logs/ models/whisper models/fasttext-langdetect
chmod 755 ./audio_capture.sh ./start.sh

# packages install
apt install python3.10-dev python3.10-venv portaudio19-dev n2n
apt install libcairo2-dev python-gi-dev libgirepository1.0-dev 
apt-get install libsdl2-dev git cmake screen ffmpeg

# install python virtual env
python3.10 -m venv .venv
source .venv/bin/activate

# python requirements install
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html

pip install faiss-gpu==1.7.2
cd .venv/lib/python3.10/site-packages/faiss
ln -s swigfaiss.py swigfaiss_avx2.py

pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install rapidocr-paddle==1.3.17

pip install -r requirements.txt

# start pulseaudio daemon to system wide
addgroup root pulse-access
/usr/bin/pulseaudio --daemonize=true --system --disallow-exit --disallow-module-loading

# download model for vector store embedding 
cd <path-to-node-app>/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/BAAI/bge-large-en-v1.5
wget https://hf-mirror.com/BAAI/bge-large-en-v1.5/resolve/main/model.safetensors?download=true
mv 'model.safetensors?download=true' bge-large-en-v1.5/model.safetensors
wget https://hf-mirror.com/BAAI/bge-large-en-v1.5/resolve/main/pytorch_model.bin?download=true
mv 'pytorch_model.bin?download=true' bge-large-en-v1.5/pytorch_model.bin

# download and compile whisper.cpp
cd /opt
git clone https://github.com/ggerganov/whisper.cpp.git
git checkout 725350d4ea1545d890fe41f815b851cbc57838f6
cd whisper.cpp
cmake -B build -DWHISPER_CUBLAS=ON -DWHISPER_SDL2=ON
cmake --build build -j --config Release
cp build/libwhisper.so <path-to-node-app>/whisper/

# download whisper ggml model
cd <path-to-node-app>/models/whisper
wget https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true
mv 'ggml-medium.bin?download=true' ggml-medium.bin
```

# Quick start running
```
# create blank database
cd <path-to-node-app>
python ./commands.py --create-tables

# start main service
python ./main.py
```

```
# start audio capture
python ./audio_capture.py
```

# Starting on boot
```
vim /etc/pulse/client.conf

# set pulse audio not to auto spawn
default-server = /var/run/pulse/native
autospawn = no
```

```
vim /lib/systemd/system/hjm-pulse.service

[Unit]
Description=Hajime bot node app pulse audio service
Requires=sound.target dbus.service
After=sound.target dbus.service

[Service]
Type=notify
ExecStart=/usr/bin/bash /opt/dorylus/pulse.sh
Restart=on-failure
KillMode=process

[Install]
WantedBy=multi-user.target

```

```
vim /etc/pulse/client.conf

default-server = /var/run/pulse/native
autospawn = no
```

MAKE SURE change the Input Source and Boost Volume setting in <path-to-node-app>/pulse.sh depending on your machine.


```
vim /lib/systemd/system/dorylus-main.service

[Unit]
Description=Dorylus Agent Engine Main
After=dorylus-pulse.service
[Service]
User=root
Type=simple
ExecStart=/bin/bash <path-to-node-app>/start.sh
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
StandardOutput=append:<path-to-node-app>/logs/main.log
StandardError=append:<path-to-node-app>/logs/main.log

[Install]
WantedBy=multi-user.target
WantedBy=graphical.target

```

```
vim /lib/systemd/system/dorylus-audio-capture.service

[Unit]
Description=Hajime bot audio capture
After=dorylus-main.service

[Service]
User=root
Type=simple
ExecStart=/usr/bin/bash <path-to-node-app>/audio_capture.sh
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
StandardOutput=append:<path-to-node-app>/logs/audio-capture.log
StandardError=append:<path-to-node-app>/logs/audio-capture.log

[Install]
WantedBy=multi-user.target
WantedBy=graphical.target

```

```
systemctl enable dorylus-pulse.service
systemctl enable dorylus-main.service
systemctl enable dorylus-audio-capture.service
```