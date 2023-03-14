import os
import json
import os.path

def save_results(opt, best_epoch=None, best_time=None, n_params=None, average_mAP=None, mAPs=None, recall=None, eval_result=None):
    """
    This function saves all the results of a run in the document history.py.
    It can be called from Train with the training information or from Eval
    to save the evaluation of the results.
    """
    
    architecture = "default"
    for a in {"use_xGPN", "use_ViT2", "use_ReMoT"}:
        if opt[a]:
            architecture = a
                        
    config={
        "run_name": opt["run_name"],
        "architecture": architecture,
        "dataset": "ego4d",
        "features": opt["features"],
        "version": "v2" if opt["clip_anno"].endswith("v2.json") else "v1",
        "batch_size": opt["batch_size"],
        "optimizer_name": "AdamW",
        "learning_rate": opt["train_lr"],
        "num_epoch": opt["num_epoch"],
        "num_heads": opt["num_heads"],
        "dim_attention": opt["dim_attention"],
        "num_levels": opt["num_levels"],
        "bb_hidden_dim": opt["bb_hidden_dim"],
        "mlp_num_hiddens": opt["mlp_num_hiddens"],
        "num_levels": opt["num_levels"],
        "mask_size": opt["mask_size"],
        "optim": opt["optim"],
        "scheduler":opt["scheduler"]
    }
    if best_epoch != None:
        config["best_epoch"]= best_epoch
    if best_time != None:
        config["best_time"]= best_time
    if n_params != None:
        config["n_params"]= n_params
    if average_mAP != None:
        config["average_mAP"]= average_mAP
        #config["mAPs"]= mAPs.tolist()
    if recall != None:
        config["recall"]= recall
        #config["eval_result"]= eval_result.tolist()
    
    
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

