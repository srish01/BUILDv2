cuda="$1"

echo "PID of this script: $$"

framework="build"
model="pass_deitadapter"
n_tasks="5"
dataset="cifar10"
adapter_latent="64"
optim="sgd"
n_epochs="20"
lr="0.005"
batch_size="64"
class_order="0"
folder="pass_cifar10-5T"

cmd="nohup python run.py --train --framework $framework --model $model --n_tasks $n_tasks --dataset $dataset --adapter_latent $adapter_latent --optim $optim --n_epochs $n_epochs --lr $lr --batch_size $batch_size --class_order $class_order --folder $folder --cuda_id $cuda"

$cmd  --kd_weight 10 --protoAug_weight 10 --temp 0.1 --use_clip_grad --clip_grad 10 &> nohups/nohup_${folder}.out &