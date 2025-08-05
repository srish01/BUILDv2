cuda="$1"
echo "Inserted cuda $cuda"
echo "PID of this script: $$"

framework="build"
model="pass_deitadapter"
n_tasks="5"
dataset="cifar10"
adapter_latent="64"
optim="sgd"
n_epochs="1"
lr="0.005"
batch_size="64"
class_order="0"
k="0.2"
p="10"
t="0.1"
clip_grad="10"
# folder="pass_cifar10-5T_k${k}_p${p}_t${t}_cg${clip_grad}"
folder="pass_debug_disperato"

cmd="nohup python run.py --train --test --framework $framework --method $framework --model $model --n_tasks $n_tasks --dataset $dataset --adapter_latent $adapter_latent --optim $optim --n_epochs $n_epochs --lr $lr --batch_size $batch_size --class_order $class_order --folder $folder --cuda_id $cuda --load_dir logs/$folder --print_filename eval.txt"

$cmd  --kd_weight $k --protoAug_weight $p --temp $t &> nohups/nohup_${folder}.out &




# nohup python run.py --test --framework build --method build --model derpp_deitadapter --n_tasks 10 --dataset cifar100 --adapter_latent 64 --test_batch_size 512 --class_order 0 --folder derpp_cifar100-10t --load_dir logs/derpp_cifar100-10t --print_filename eval_derpp.txt --cuda_id 1 > nohups/nh_derpp_c100_10t.out &
# nohup python run.py --test --framework build --method build --model derpp_deitadapter --n_tasks 20 --dataset cifar100 --adapter_latent 64 --test_batch_size 512 --class_order 0 --folder derpp_cifar100-20t --load_dir logs/derpp_cifar100-20t --print_filename eval_derpp.txt --cuda_id 1 > nohups/nh_derpp_c100_20t.out &

# 3869964
# 3869444