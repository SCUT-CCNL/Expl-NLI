 This code is for paper "Explainable Natural Language Inference via Identifying Important Rationales". 
 
 1. To run this code, you should obtain the data from https://github.com/OanaMariaCamburu/e-SNLI, then process it by run the process_data.py;
 2. go to the file of 'rationale_extra' and run the train.py and then the predict.py to predict the rationales;
 3. go to the file of 'rationales_sel' and run the main_selector.py to selector the rationales;
 4. go to the file of 'generator' and 
   4.1 finetune the GPT2 model: run the 'prepare_data_for_finetune.py' and 'GPT2_finetune_lm.py';
   4.2 generate the NLEs by the finetuned model: run the 'prepare_data_for_generation.py' and 'GPT2_generate.py';
 5. go to the file of 'classify', and run the 'main_classify.py' to obtain the prediction results.
 
Note: All parameters use default values, and you can also set them at runtime.
