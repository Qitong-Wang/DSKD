echo "-------------------------"
echo "Gather Sense Embeddings from Wiki"
echo "-------------------------"


for i in 1 2 3 4 5 6
do
mkdir -p ./EMB_llama/wiki/
time torchrun   --nproc_per_node=6 --nnodes=1  main.py --world_size 6 --save_dir ./EMB_llama/wiki/ --model_name llama --task_name wiki$1  --eval_json_path ./dataset/wiki_split/$1.json  --run_option gather     
done

python a0_generate_combinepkl_withsplit.py --folder_names ./EMB_llama/wiki/ --output_path ./EMB_llama/combine

echo "-------------------------"
echo "Running Gather  for dataset"
echo "-------------------------"

time torchrun   --nproc_per_node=6 --nnodes=1 main.py  --world_size 6 --save_dir ./EMB_llama/embedding/ --task_name squad  --model_name  llama --run_option gather --gather_size 100  --eval_json_path  ./dataset/SQuADV2/train.json  --nshot 1 

for datasetname in mmlu csqa piqa arc
do 

task_name=${datasetname}_train
time torchrun   --nproc_per_node=6 --nnodes=1  main.py --world_size 6 --eval_json_path None --gather_size 100 --model_name llama --task_name $task_name --save_dir ./EMB_llama/embedding/  --nshot 5   --run_option gather 

done


python pickle_utils.py -f ./EMB_llama/embedding/ -o ./EMB_llama/train.combinepkl -m dict_nparray_withfilter  --filter 2000

echo "-------------------------"
echo "Combine Train and Wiki"
echo "-------------------------"

for i in 1 2 3 
do

python a3_add_combinepkl.py --base_file ./EMB_llama/combine.part${i}.combinepkl --add_file ./EMB_llama/train.combinepkl --add_file_num 2000 --output_path ./EMB_llama/combine.part${i}.combinepkl --add_record_txt ./output_txt/add_train_part${i}.txt

done 

python a2_remain_addfile.py --add_file ./EMB_llama/train.combinepkl --output_path ./EMB_llama/combine.part4.combinepkl --record_txt ./output_txt/add_train_part1.txt ./output_txt/add_train_part2.txt ./output_txt/add_train_part3.txt

echo "-------------------------"
echo "Running Cluster Embedding"
echo "-------------------------"


for i in 1 2 3 4
do
python a3_kmeans_lasttoken.py --input_path ./EMB_llama/combine.part${i}.combinepkl --output_path ./EMB_llama/sensedict/sensedict_part${i}.dictpkl  --k 10

done

python pickle_utils.py -f ./EMB_llama/sensedict/ -o ./EMB_llama/sensedict.dictpkl -m dict_replace


python a4_word_dict.py --syn_pkl ./relationship/llama_base_syn.pkl --ant_pkl ./relationship/llama_base_ant.pkl --sense_pkl ./EMB_llama/combine.kmeanspkl --out_syn_vid ./relationship/llama_syn.pkl --out_ant_vid ./relationship/llama_ant.pkl --out_cloud_dict ./EMB_llama/combine.dictpkl --m 3 --K 10 


