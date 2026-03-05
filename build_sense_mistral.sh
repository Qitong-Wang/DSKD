echo "-------------------------"
echo "Gather Sense Embeddings from Wiki"
echo "-------------------------"


for i in 1 2 3 4 5 6
do
mkdir -p ./EMB_mistral/wiki/
time torchrun   --nproc_per_node=6 --nnodes=1  main.py --world_size 6 --save_dir ./EMB_mistral/wiki/ --model_name mistral --task_name wiki$1  --eval_json_path ./dataset/wiki_split/$1.json  --run_option gather     
done

python a0_generate_combinepkl_withsplit.py --folder_names ./EMB_mistral/wiki/ --output_path ./EMB_mistral/combine

echo "-------------------------"
echo "Running Gather  for dataset"
echo "-------------------------"

time torchrun   --nproc_per_node=6 --nnodes=1 main.py  --world_size 6 --save_dir ./EMB_mistral/embedding/ --task_name squad  --model_name  mistral --run_option gather --gather_size 100  --eval_json_path  ./dataset/SQuADV2/train.json  --nshot 1 

for datasetname in mmlu csqa piqa arc
do 

task_name=${datasetname}_train
time torchrun   --nproc_per_node=6 --nnodes=1  main.py --world_size 6 --eval_json_path None --gather_size 100 --model_name mistral --task_name $task_name --save_dir ./EMB_mistral/embedding/  --nshot 5   --run_option gather 

done


python pickle_utils.py -f ./EMB_mistral/embedding/ -o ./EMB_mistral/train.combinepkl -m dict_nparray_withfilter  --filter 2000

echo "-------------------------"
echo "Combine Train and Wiki"
echo "-------------------------"

for i in 1 2 3 
do

python a3_add_combinepkl.py --base_file ./EMB_mistral/combine.part${i}.combinepkl --add_file ./EMB_mistral/train.combinepkl --add_file_num 2000 --output_path ./EMB_mistral/combine.part${i}.combinepkl --add_record_txt ./output_txt/add_train_part${i}.txt

done 

python a2_remain_addfile.py --add_file ./EMB_mistral/train.combinepkl --output_path ./EMB_mistral/combine.part4.combinepkl --record_txt ./output_txt/add_train_part1.txt ./output_txt/add_train_part2.txt ./output_txt/add_train_part3.txt

echo "-------------------------"
echo "Running Cluster Embedding"
echo "-------------------------"


for i in 1 2 3 4
do
python a3_kmeans_lasttoken.py --input_path ./EMB_mistral/combine.part${i}.combinepkl --output_path ./EMB_mistral/sensedict/sensedict_part${i}.dictpkl  --k 5

done

python pickle_utils.py -f ./EMB_mistral/sensedict/ -o ./EMB_mistral/sensedict.dictpkl -m dict_replace


python a4_word_dict.py --syn_pkl ./relationship/mistral_base_syn.pkl --ant_pkl ./relationship/mistral_base_ant.pkl --sense_pkl ./EMB_mistral/combine.kmeanspkl --out_syn_vid ./relationship/mistral_syn.pkl --out_ant_vid ./relationship/mistral_ant.pkl --out_cloud_dict ./EMB_mistral/combine.dictpkl --m 3 --K 5


