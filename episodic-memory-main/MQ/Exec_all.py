# nuevo código para hacer train infer y eval de golpe

import sys
import wandb
sys.dont_write_bytecode = True
import os
import json
import os.path
import torch
import Utils.opts as opts
import datetime


from Train import Train_VSGN
from Infer import Infer_SegTAD

from Evaluation.ego4d.generate_detection import gen_detection_multicore as gen_det_ego4d
from Evaluation.ego4d.get_detect_performance import evaluation_detection as eval_det_ego4d
from Evaluation.ego4d.generate_retrieval import gen_retrieval_multicore as gen_retrieval
from Evaluation.ego4d.get_retrieval_performance import evaluation_retrieval as eval_retrieval



torch.manual_seed(21)

if __name__ == '__main__':

    print(datetime.datetime.now())

    opt = opts.parse_opt()
    opt = vars(opt)

    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
        
    if not os.path.exists(opt["output_path"]):
        os.makedirs(opt["output_path"])

    print(opt)

    print("---------------------------------------------------------------------------------------------")
    print("Training starts!")
    print("---------------------------------------------------------------------------------------------")
    
    name1 = "Train_batch:" + str(opt["batch_size"]) + "_lr:" + str(opt["train_lr"])
    
    if opt["wandb"] == "true":
        wandb.login()
        wandb.init(
            project="Ego4d default",
            name=name1,
            config={
                "architecture": "VSGN",
                "dataset": "ego4d",
                "batch_size": opt["batch_size"],
                "optimizer_name": "Adam",
                "learning_rate": opt["train_lr"],
                "num_epoch": opt["num_epoch"],
                "use_xGPN": opt["use_xGPN"],
                "use_ViT": opt["use_ViT"]
            })

    
    Train_VSGN(opt)
    print("Training finishes!")
    
    if opt["wandb"] == "true":
        wandb.finish()

    print(datetime.datetime.now())

    print(datetime.datetime.now())
    print("---------------------------------------------------------------------------------------------")
    print("1. Inference starts!")
    print("---------------------------------------------------------------------------------------------")

    Infer_SegTAD(opt)

    print("Inference finishes! \n")
    
    
    average_mAP, recall = 0, 0 

    if opt['eval_stage'] == 'eval_detection' or opt['eval_stage'] == 'all':
        print("---------------------------------------------------------------------------------------------")
        print("2. Detection evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Generate detections!")
        gen_det_ego4d(opt)   # Not knowing video categories

        if 'val' in opt['infer_datasplit']:
            print("b. Evaluate the detection results!")
            average_mAP = eval_det_ego4d(opt)
            print("Detection evaluation finishes! \n")
            

    if opt['eval_stage'] == 'eval_retrieval' or opt['eval_stage'] == 'all':

        print("---------------------------------------------------------------------------------------------")
        print("3. Retrieval evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Generate retrieval!")
        gen_retrieval(opt)

        if 'val' in opt['infer_datasplit']:
            print("b. Evaluate the retrieval results!")
            recall = eval_retrieval(opt)
            print("Detection evaluation finishes! \n")
            
            
    config={
        "architecture": "VSGN_default",
        "dataset": "ego4d",
        "batch_size": opt["batch_size"],
        "optimizer_name": "Adam",
        "learning_rate": opt["train_lr"],
        "num_epoch": opt["num_epoch"],
        "use_xGPN": opt["use_xGPN"],
        "use_ViT": opt["use_ViT"],
        "average_mAP": average_mAP,
        "recall": recall
    }
    
    history_path = opt["history_path"]
    if not os.path.exists(history_path):
        data = {0: config}
        with open(history_path, "w") as f:
            json.dump(data, f)
    else:
        with open(history_path, "r") as f:
            data = json.load(f)
        data[len(data)] = config
        with open(history_path, "w") as f:
            json.dump(data, f)
