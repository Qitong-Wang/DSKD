export PYTHONPATH=$(pwd)


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

 

echo "#####################################################################################"
echo "Train"  
echo "#####################################################################################"

mkdir -p ./ckpt/

time torchrun     --nproc_per_node=3 --nnodes=1  main.py --world_size 3 --teacher_device_start_indice 3 --save_dir ./ckpt/  --train_json_path ./dataset/SQuADV2/train_03.json   --task_name all --model_name mistral --sense_dict_path ./EMB_mistral/sensedict_relationship.dictpkl --lr 0.00002 --epoch 2 --ckpt_path ./ckpt/mistral_ --run_option train --student_total_layer 16 --student_train_layer 2 --nshot 5 --loss_type 1 --sense_weight 1.5  --kappa 5  --syn_dict_path ./relationship/mistral_syn.pkl --ant_dict_path ./relationship/mistral_ant.pkl 


for datasetname in arc csqa piqa  mmlu
do 
if [ "$datasetname" = "mmlu" ]; then
    task_name="${datasetname}_test"
 
else
    task_name="${datasetname}_validation"
fi

echo "#####################################################################################"
echo "Evaluation Task name: " $task_name 
echo "#####################################################################################"

time torchrun   --nproc_per_node=6 --nnodes=1 main.py  --world_size 6  --task_name $task_name  --run_option student --model_name mistral   --nshot 5 --student_total_layer 16 --load_student_ckpt_path ./ckpt/mistral_1.pth 

done

echo "#####################################################################################"
echo "Evaluation Task name: squad"
echo "#####################################################################################"

time torchrun   --nproc_per_node=6 --nnodes=1  main.py  --world_size 6 --task_name squad  --model_name  mistral   --run_option student --eval_json_path  ./dataset/SQuADV2/dev.json  --nshot 1 --student_total_layer 16  --load_student_ckpt_path ./ckpt/mistral_1.pth 



