# BUILD

This repository is the official implementation of BUILD: Buffer-free Incremental Learning with OOD Detection for the Wild



# Pre-trained transformer

https://drive.google.com/file/d/1uEpqe6xo--8jdpOgR_YX3JqTHqj34oX6/view?usp=sharing

and save the file as ./deit_pretrained/best_checkpoint.pth



# Requirements
Please install a virtual environment

```
conda create -n more python=3.8 anaconda
```

Activate the environment

```
conda activate more
```

Please install the following packages in the environment

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install ftfy
pip install timm==0.4.12
```

# Data
Before performing training or evaluation function, download datasets: cifar10, cifar100, tinyimagenet in the default directory as `./data` folder. After downloading TinyImageNet, use `bash tinyimagenet.sh` to assign labels to the images in val dataset. 
	
# Testing
e.g., For Cifar100-10T,

# Training
train using build framework
```
python run.py --train --framework build --model deitadapter_more --n_tasks 10 --dataset timgnet --adapter_latent 64 --optim sgd --n_epochs 40 --lr 0.001 --batch_size 64 --class_order 0 --folder build_cifar100-10T
```

# Testing

## testing BUILD models

python run.py --test --framework build --method build --model deitadapter_more --load_dir logs/build_cifar100-10T --dataset cifar100 --test_batch_size 32 --adapter_latent 64 --optim sgd --folder build_cifar100-10T --print_filename eval_build.txt --n_tasks 10

## testing MORE models

python run.py --test --framework build --method more_fw --model deitadapter_more --load_dir logs/more_cifar100-10T --dataset cifar100 --test_batch_size 32 --adapter_latent 64 --optim sgd --folder more_cifar100-10T --print_filename eval_more-fw.txt --n_tasks 10

- for testing models after forward pass, method: more_fw
- for testing models after backward pass, method: more_bw

# For MORE implementation we keep their original script with minor edits

## For Forward pass
python run.py --framework more --model deitadapter_more --n_tasks 10 --dataset cifar100 --adapter_latent 64 --optim sgd --compute_md --compute_auc --buffer_size 2000 --n_epochs 40 --lr 0.001 --batch_size 64 --use_buffer --class_order 0 --folder more_cifar100-10T

## For Backward pass:

python run.py --framework more --model deitadapter_more --n_tasks 10 --dataset cifar100 --adapter_latent 64 --optim sgd --compute_auc --buffer_size 2000 --folder more_cifar100-10T --load_dir logs/more_cifar100-10T --n_epochs 10 --print_filename model_backupdate.txt --use_buffer --load_task_id 19 --train_clf --train_clf_save_name model_backupdate --class_order 0


# Acknowledgement
The code format follows DER++ and HAT. A large chunk of the codes is a direct modiciation of MORE and HAT, pytorch, timm, numpy, and sklearn.

https://github.com/aimagelab/mammoth

https://github.com/joansj/hat