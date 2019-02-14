# object_detection

This repository is forked and modified from tensorflow object detection model. 

To run baseline code for the hackathon, you may follow the instruction below:

1) # fork / mirror this repository      
    !!!important: make it as your own private repository and assign right to YITU admin （refer to section 5 in "Hackathon_Baseline_User_Guider"!!! 
    

2) # environment setttings

    ``` bash
    # activate AWS virtual environment
    cd ~
    source activate tensorflow_p36
    ```

    ``` bash
    # Setup PYTHONPATH
    cd hackathon-sg/ 
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim    
    ```

3) # to generate tfrecord data

    ``` bash
    cd object_detection
    python script/generate_tfrecord_from_csv.py --image_dir ../input/training/images/ --output_path ../input/hackathon --csv_file ../input/training/train_label.csv --validation_set_size 500
    ```

4) # to train the model 

    ``` bash 
    python script/train.py --logtostderr --train_dir=training/baseline/ --pipeline_config_path=training/hackathon_baseline.config
    ```
   #to visualize the training results, can only use 8000 for port no. 
      tensorboard --logdir=training/baseline --port 8000
     
5) # to eval the trained model 

    ``` bash 
    python script/eval.py --logtostderr --pipeline_config_path=training/hackathon_baseline.config --checkpoint_dir=training/baseline --eval_dir=eval/baseline
    ```
    
   #To visualize the eval results, can only use 8000 for port no. 
      tensorboard --logdir=eval/baseline --port 8000

6) # to export the trained model 

    ``` bash
    python script/export_inference_graph.py --input_type image_tensor --pipeline_config_path training/hackathon_baseline.config   --trained_checkpoint_prefix training/baseline/model.ckpt-20000 --output_directory output/
    ```
    
7) # to output the results to .csv

    ``` bash
    python script/output_csv_results.py threshold=0.5 data_dir=../input/testing/images/ model_path=output/frozen_inference_graph.pb output_path=output/submission.csv label_map=../input/label_map.pbtxt
    ```
        
8) # to submit the results 
    please follow the instruction from ”Hackathon Infra User Manual“ to submit your detection results. 

    git push your modified codes to your private repository whenever you make a submission.
    
    !!!DO PUSH YOUR CODES FOR EVERY SUBMISSION YOU MADE FOR CODE VERIFICATION PURPOSE!!!



    
 
