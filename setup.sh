conda remove --name DiffME --all -y
conda create -n DiffME python=3.8 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia