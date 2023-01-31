This is the folder dedicated to the Moment Queries challenge. 

###Take a look at the files:
- In this folder we find the files to run, such as Train.py, Eval.py, ... and history.json to store the results.
- Inside Evaluation there is the information about the annotations and the functions of the evaluation metrics.
- In Models, there are the classes of all the models executable by the program and their classes, including ViT2 (which executes the algorithm called ViT+ in memory) and ReMoT.
- In Utils there are complementary functions, such as the definition of the opts, which determines the parameters with which we can execute the code, or save.py that is used to save the results.


### Environment Installation
To run all the files you will need to have the following packages installed:
```
    conda create -n pytorch160 python=3.7 
    conda activate pytorch160   
    conda install pytorch=1.6.0 torchvision cudatoolkit=10.1.243 -c pytorch   
    conda install -c anaconda pandas    
    conda install -c anaconda h5py  
    conda install -c anaconda scipy 
    conda install -c conda-forge tensorboardx   
    conda install -c anaconda joblib    
    conda install -c conda-forge matplotlib 
    conda install -c conda-forge urllib3
    pip install d2l==1.0.0b0
    pip install wandb
```
### Data required
To run the code you need the Ego4D videos or features and the relevant annotations. Go to https://ego4d-data.org/ for more information.


### Annotation conversion 
First of all, the video annotations have to be converted. To do this, use the following instruction.
```
    python Convert_annotation.py
```

!!! You have to change all the paths of the documents and instructions to those of your own machine, now they are all set to those used by the author for the execution in Peregrine.


The following instructions can be used for training, prediction and evaluation.

### Training
```    
     python Train.py --use_ReMoT --is_train true --dataset ego4d --feature_path {DATA_PATH} --checkpoint_path {CHECKPOINT_PATH} --batch_size 32 --train_lr 0.0001
```
### Inference
```
     python Infer.py --use_ReMoT --is_train false --dataset ego4d --feature_path {DATA_PATH} --checkpoint_path {CHECKPOINT_PATH}  --output_path {OUTPUT_PATH}   
```
### Evaluation
```
     python Eval.py --dataset ego4d --output_path {OUTPUT_PATH} --out_prop_map {OUT_PMAP} --eval_stage all
```

It is important that all 3 instructions use the same parameters to save the results of the correct execution.

Use the --testing option to see the shapes of the data at all moments, great to debug the code.

In the scripts.sh document there is an example of instructions we used to run tests in Peregrine.



## Acknowledgements

This codebase is built on  [Ego4D](https://github.com/EGO4D/episodic-memory/tree/main/MQ).

