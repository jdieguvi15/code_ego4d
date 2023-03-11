from Evaluation.ego4d.generate_detection import gen_detection_multicore as gen_det_ego4d
from Evaluation.ego4d.get_detect_performance import evaluation_detection as eval_det_ego4d
from Evaluation.ego4d.generate_retrieval import gen_retrieval_multicore as gen_retrieval
from Evaluation.ego4d.get_retrieval_performance import evaluation_retrieval as eval_retrieval

import Utils.opts as opts
from Utils.save import save_results
import os.path
if __name__ == '__main__':

    opt = opts.parse_opt()
    opt = vars(opt)
    opt = opts.improve_opt(opt)

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
            mAPs, average_mAP = eval_det_ego4d(opt)
            print("Detection evaluation finishes! \n")
            

    if opt['eval_stage'] == 'eval_retrieval' or opt['eval_stage'] == 'all':

        print("---------------------------------------------------------------------------------------------")
        print("3. Retrieval evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Generate retrieval!")
        gen_retrieval(opt)

        if 'val' in opt['infer_datasplit']:
            print("b. Evaluate the retrieval results!")
            recall, eval_result = eval_retrieval(opt)
            print("Detection evaluation finishes! \n")
            
    save_results(opt, average_mAP=average_mAP, mAPs=mAPs, recall=recall, eval_result=eval_result)
