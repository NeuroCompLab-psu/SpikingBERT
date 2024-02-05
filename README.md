SpikingBERT
======== 
SpikingBERT is a Spiking variant of BERT based encoder-only language model architecture. Both internal layer and prediction layer Knowledge Distillation (KD) is performed
to improve model performance. Moreover, KD is performed on both general data (pre-training phase) as well as task-specific KD (i.e. for finetuning).
SpikingBERT being a spiking architecture allows for dynamic tradeoff of energy/power consumption and accuracy of the model. Moreover, the KD techniques deployed
allows to reduce number of model parameters.



This code implements the methodology described in the paper titled: "SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit
Differentiation".

Installation
============
Run command below to install the required packages (**using python3**)
```bash
pip install -r requirements.txt
```

STEP 1: General KD (BERT<sub>BASE</sub>) to SpikingBERT
====================
In this step we use a general domain data such as wikitext-103, wikitext-2 which we process and use to perform KD from a general BERT<sub>BASE</sub> model (uncased) to our SpikingBERT model following the steps given below.

(a) Download a pre-trained bert-base-uncased-model from Huggingface or pre-train a BERT model from scratch.  Create
a configuration file similar to that in spiking_student_model folder (given as a reference).

(b) Download the general corpus and use `preprocess_training_data.py` (an existing pre-processing code) to produce the corpus in json format  


```
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.
 
python preprocess_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$                              
```





(c) Use `spiking_bert_general_distill.py` to run the general KD. Pass the directory location of the processed corpus, pre-trained BERT model and spiking student config as parameters. A sample spiking student config is added as part of the project in folder spiking-student-model.\
Note: Since the code uses DataParallel, use CUDA_VISIBLE_DEVICES to specify the GPUs.

``` 
python spiking_bert_general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE_DIR}$ \
                          --spiking_student_model ${SPIKING_STUDENT_CONFIG_DIR}$ \
                          --do_lower_case \
                          --train_batch_size 128 \
                          --t_conv $t_conv$ \
                          --vth $vth$ \
                          --output_dir ${GENERAL_SPIKINGBERT_DIR}$ 
```

Step 2: Task-based Internal layer KD (IKD) from Finetuned BERT to task-specific SpikingBERT
==========================
In the task-specific distillation, we perform task-based IKD as described in the paper to create a corresponding task-specific version of SpikingBERT.

(a) Download datasets from GLUE benchmark. You can directly download the datasets from GLUE website under tasks section and unzip all the relevant tasks described in the paper in a separate directory (named for example: glue_benchmark). Or use publicly available script given in this project: download_glue_data.py

(b) Download corresponding fine-tuned BERT models (specific to each dataset like SST-2, QQP, MNLI, STS-B, etc.) from Huggingface (like bert-base-uncased-SST-2, bert-base-uncased-QQP, bert-base-uncased-MNLI, etc.) or fine-tune an existing BERT model.

(c): Use `spiking_bert_task_distill.py` to run task-based IKD. The spiking_student_model can be the directory where the output of the General KD phase is stored. \
Note: Since the code uses DataParallel, use CUDA_VISIBLE_DEVICES to specify the GPUs.

```

# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.

python spiking_bert_task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --spiking_student_model ${GKD_SPIKINGBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${IKD_SPIKINGBERT_DIR}$ \
                       --max_seq_length 128 \
                       --t_conv $t_conv$ \
                       --vth $vth$ \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --do_lower_case  
                         
```



Step 3: Task-specific Prediction layer KD (IKD) from Finetuned BERT to SpikingBERT
==========================
(a) use `spiking_bert_task_distill.py` with flag --pred_distill to run the prediction layer distillation. Either in place of or post prediction-layer distillation we can also finetune the model further by adding the flag --train_true_labels
which will allow us to train the model not using the output logits of a finetuned BERT model but the actual true labels of the samples used. The spiking_student_model can be the directory where the output of the task-based IKD phase is stored.\
Note: \
(1) Since the code uses DataParallel, use CUDA_VISIBLE_DEVICES to specify the GPUs.\
(2) Use flag --train_true_labels to train the SpikingBERT model using actual true labels (instead of prediction layer distillation i.e. using logits of trained BERT model).


```

python spiking_bert_task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --spiking_student_model ${IKD_SPIKINGBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --max_seq_length 128 \
                       --output_dir ${FINAL_SPIKINGBERT_DIR}$ \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --t_conv $t_conv$ \
                       --vth $vth$ \
                       --train_batch_size 32 
                       
```
The hyper-parameters given in the paper can be used to recreate the results. Since the model size is  greater than 50 MB (size limit of submission) we could not upload a finetuned model for direct evaluation. However, we added logs for all 3 STEPS for different scenarios like general KD, task-based IKD and pred_layer distill. 
It is to be noted that for pred_distill 2-4 epochs are enough for almost all tasks once task-based IKD is done properly.


Evaluation
==========================
The `spiking_bert_task_distill.py` also provide the evalution by adding the flag --do_eval:

```
${FINAL_SPIKINGBERT_DIR}$ includes the final trained student spiking model along with config file and vocab file.

python spiking_bert_task_distill.py --do_eval \
                       --spiking_student_model ${FINAL_SPIKINGBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${EVAL_SPIKINGBERT_DIR}$ \
                       --max_seq_length 128 \ 
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --t_conv $t_conv$ \
                       --vth $vth$ 
                                   
```
File Details
==========================
(1) models/snn_ide_bert* : These files define the high-level structure of the BERT model. The *classi* files are used for task-based IKD and final pred-distillation and the other one for general KD. The number 2 and 4 signifies 2 and 4 spiking encoder layers respectively.\
(2) modules/snn_modules.py : SNNBERTSpikingLIFFuncMultiLayer class contains the spiking operations inside the model. The neuron dynamics and spike generation are all part of this class. The model is operated for Tconv time steps as described in the paper.\
(3) modules/snnide_bert_multilayer_modules.py : SNNIDEBERTSpikingMultiLayerModule contains code for efficient training using the method described in the paper.\
(4) modules/snn_bert_modules.py : Code for individual components specified in the paper and appendix.\
(5) transformer: This folder contains code specific to BERT (modelling, utilities, optimizer, etc.).

## Table of important hyper-params and  command line arguments. 

|Arguments|Description|Examples|
|-------|------|------|
|`max_seq_length`|Maximum Sequence length.|`--max_seq_length 128`|
|`t_conv`|Time steps for convergence.|`--t_conv 125`|
|`Vth`|Threshold voltage.|`--vth 1.0`|
|`train_batch_size`|Batch size for training.|`--train_batch_size 32`|
|`eval_batch_size`|Batch size for evaluation.|`--eval_batch_size 32`|
|`learning_rate`|Learning rates.|`--learning_rate 3e-5`|
|`temperature`|CE temperature during pred_distill.|`--temperature 1.0`|
|`num_train_epochs`|Number of epochs for training.|`--num_train_epochs 5`|
|`epochs`|Number of epochs.|`--epochs 200`|
|`do_eval`|Perform evaluation.|`--do_eval`|
|`seed`|Random seed select.|`--seed 0`|
|`do_lower_case`|convert words to lower case.|`--do_lower_case`|
|`spiking_student_model`|Spiking student model directory path.|`--spiking_student_model ${Path}$`|
|`teacher_model`|Teacher model directory path.|`--teacher_model ${Path}$`|
|`output_dir`|Output directory path.|`--output_dir ${Path}$`|
|`task_name`|Name of the dataset.|`--task_name QQP`|
|`data_dir`|Directory of the dataset.|`--data_dir ./glue_benchmark/SST-2/`|


