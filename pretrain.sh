python run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=8 ++exp.runner.total_steps=70000 \
  +model=masked_tf_model_large ++model.input_dim=40 \
  +data=masked_spec +data.data=/mnt/AI_Magic/projects/iEEG_data/ucla/EEG_Data_90min/ ++data.val_split=0.1 ++data.test_split=0.1 +preprocessor=stft \
  +criterion=pretrain_masked_criterion \
  +task=fixed_mask_pretrain.yaml ++task.freq_mask_p=0.05 ++task.time_mask_p=0.05 
