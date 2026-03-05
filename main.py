# main.py
import os
import argparse
from train import train
from evaluate import evaluate
from inference import inference 

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_json_path', type=str, default="None")
    p.add_argument('--eval_json_path', type=str, default="None")
    p.add_argument('--sense_dict_path', type=str, default="./sense_tensor/teacher.kmeanspkl")
    p.add_argument('--log_file_path', type=str, default="./log.txt")
    p.add_argument('--run_option', type=str, default="original",
                  help="train | original | inference | student | "
                       "gather ")
    p.add_argument('--save_dir', type=str, default="./encoding/")
    p.add_argument('--task_name', type=str, default="anatomy")
    p.add_argument('--world_size', type=int, default=6)
    p.add_argument('--nshot', type=int, default=1)


    p.add_argument('--model_name', type=str, default="llama")
    p.add_argument('--student_total_layer', type=int, default=16)
    p.add_argument('--student_train_layer', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--ckpt_path', type=str, default="./ckpt/model")
    p.add_argument('--load_student_ckpt_path', type=str, default="empty")
    p.add_argument('--loss_type', type=int, default=1)
    p.add_argument('--teacher_device_start_indice', type=int, default=3)
    p.add_argument('--sense_weight', type=float, default=1.0)


    p.add_argument('--gather_size', type=int, default=50)
    p.add_argument('--sample_size', type=float, default=0.4)
    p.add_argument('--kappa', type=int, default=1)
    p.add_argument('--neighbors', type=int, default=25)
    p.add_argument('--syn_dict_path', type=str, default="./syn_dict/syn.pkl")
    p.add_argument('--ant_dict_path', type=str, default="./syn_dict/ant.pkl")
    p.add_argument('--hinge', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--prompt_type', type=int, default=0)
    return p

def main():
    # keep tokenizer workers quiet unless user overrides
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = build_parser()
    args = parser.parse_args()
    print(args)

    if args.run_option == "train":
        train(args)
    elif args.run_option == "inference":
        inference(args)
    else:
        evaluate(args)

if __name__ == "__main__":
    main()
