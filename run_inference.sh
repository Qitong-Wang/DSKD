export PYTHONPATH=$(pwd)


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


for datasetname in arc  csqa piqa  mmlu
do 
if [ "$datasetname" = "mmlu" ]; then
    task_name="${datasetname}_test"
 
else
    task_name="${datasetname}_validation"
fi



time torchrun   --nproc_per_node=6 --nnodes=1  main.py  --world_size 6  --task_name $task_name  --run_option inference --model_name llama  --nshot 5 --student_total_layer 16 --load_student_ckpt_path ./llama-dskd 

time torchrun   --nproc_per_node=6 --nnodes=1  main.py  --world_size 6  --task_name $task_name  --run_option inference --model_name mistral  --nshot 5 --student_total_layer 16 --load_student_ckpt_path ./mistral-dskd   

done

time torchrun   --nproc_per_node=6 --nnodes=1  main.py  --world_size 6  --task_name squad --eval_json_path  ./dataset/SQuADV2/dev.json   --run_option inference --model_name llama  --nshot 1 --student_total_layer 16 --load_student_ckpt_path ./llama-dskd

time torchrun   --nproc_per_node=6 --nnodes=1  main.py  --world_size 6  --task_name squad --eval_json_path  ./dataset/SQuADV2/dev.json   --run_option inference --model_name mistral  --nshot 1 --student_total_layer 16 --load_student_ckpt_path ./mistral-dskd  
