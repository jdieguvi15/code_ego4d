from Evaluation.ego4d.generate_detection import gen_detection_multicore as gen_det_ego4d
from Evaluation.ego4d.get_detect_performance import evaluation_detection as eval_det_ego4d
from Evaluation.ego4d.generate_retrieval import gen_retrieval_multicore as gen_retrieval
from Evaluation.ego4d.get_retrieval_performance import evaluation_retrieval as eval_retrieval

import Utils.opts as opts
import os
import json
import os.path
if __name__ == '__main__':

    opt = opts.parse_opt()
    opt = vars(opt)

    print(opt)

    if not os.path.exists(opt["output_path"]):
        print('No predictions! Please run inference first!')

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
            
    architecture = "???"
    for a in {"use_xGPN", "use_ViT", "use_ViT2", "use_ViTFeatures", "use_Transformer"}:
        if opt[a]:
            architecture = a
            
    config={
        "run_name": opt["run_name"],
        "architecture": architecture,
        "dataset": "ego4d",
        "batch_size": opt["batch_size"],
        "optimizer_name": "Adam",
        "learning_rate": opt["train_lr"],
        "num_epoch": opt["num_epoch"],
        "use_xGPN": opt["use_xGPN"],
        "use_ViT": opt["use_ViT"],
        "num_heads": opt["num_heads"],
        "dim_attention": opt["dim_attention"],
        "num_levels": opt["num_levels"],
        "bb_hidden_dim": opt["bb_hidden_dim"],
        "mlp_num_hiddens": opt["mlp_num_hiddens"],
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
