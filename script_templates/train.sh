cd /home/sonlowami/carnegie-mellon/internship/HealthASR/training/nemo
python main.py 
    --config=/home/sonlowami/carnegie-mellon/internship/HealthASR/config/nemo_train.yaml 
    --train_schema=/home/sonlowami/carnegie-mellon/internship/HealthASR/data_cleaning/outputs/features/kidawida/train_features.tsv 
    --val_schema=/home/sonlowami/carnegie-mellon/internship/HealthASR/data_cleaning/outputs/features/kidawida/dev_features.tsv
    --feature_base_dir=/home/sonlowami/carnegie-mellon/internship/HealthASR/data_cleaning/outputs/features/kidawida
    --feature_key=feature_path
    --text_key=transcript
    --tokenizer_dir=/home/sonlowami/carnegie-mellon/internship/HealthASR/tokenizers/kidawida/tokenizer_spe_bpe_v1024
    --num_workers=2
    --model_class=nemo.collections.asr.models.ASRModel
    --pretrained_model="mbazaNLP/stt_rw_sw_lg_conformer_ctc_large"