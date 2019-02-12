# object_detection

This repository is forked and modified from tensorflow object detection model. 

To run baseline code for the hachathon, you may follow the instruction below:

1) # fork / clone this repository 
    fork https://github.com/Dawdleryang/object_detection.git 
    ！！！ important: make it as your own private repository and assign right to YITU admin 
    
    ``` bash
    # cd hackathon_sg
    git clone https://github.com/your_own_private_repository 
    ```

    ``` bash
    # From hackathon_sg/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```
   

2) # to generate tfrecord data

    ``` bash
    python script/generate_tfrecord_from_csv.py --image_dir ../input/training/images/ --output_path ../input/yitu --csv_file ../input/training/train_label.csv --validation_set_size 500
    ```

3) # to train the model 

    ``` bash 
    python script/train.py --logtostderr --train_dir=training/baseline/ --pipeline_config_path=training/baseline.config
    ```
   #to visualize the training results
      tensorboard --logdir=training/baseline
     
4) # to eval the trained model 

    ``` bash 
    python script/eval.py --logtostderr --pipeline_config_path=training/baseline.config --checkpoint_dir=training/baseline --eval_dir=training/baseline
    ```
    
   #To visualize the eval results
      tensorboard --logdir=eval/

5) # to export the trained model 

    ``` bash
    python script/export_inference_graph.py --input_type image_tensor --pipeline_config_path training/baseline.config   --trained_checkpoint_prefix training/baseline/model.ckpt-20000 --output_directory output/
    ```
    
6) # to output the results to .csv

    ``` bash
    python script/output_csv_results.py threshold=0.5 data_path=../input/testing/images/ model_path=output/frozen_inference_graph.pb output_path=output/submission.csv label_map=../input/label_map.pbtxt
    ```
        
7) # to submit the results 
    please follow the instruction from this link

    !!!DO PUSH YOUR CODES FOR EVERY SUBMISSION YOU MADE FOR CODE VERIFICATION PURPOSE!!!



    
 
