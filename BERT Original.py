import tokenizers
bwpt = tokenizers.BertWordPieceTokenizer(
    vocab_file=None,
    add_special_tokens=True,
    unk_token='[UNK]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
    wordpieces_prefix='##'
)


bwpt.train(
    files=["../input/custom-corpus/clean.txt"],
    vocab_size=30000,
    min_frequency=3,
    limit_alphabet=1000,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]']
)


bwpt.save(".../working/", "English")

!python create_pretraining_data.py \
    --input_file=/../input/custom-corpus/clean.txt \
    --output_file=.../working/tf_examples.tfrecord \
    --vocab_file=.../working/English-vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=42 \
    --dupe_factor=5

!python run_pretraining.py \
    --input_file=gs://tf-large-model/*.tfrecord \
    --output_dir=gs://tf-large-model/model/ \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=.../input/bert-large-uncased/config.json \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=20 \
    --num_warmup_steps=10 \
    --learning_rate=2e-5 \
    --use_tpu=True \
    --tpu_name=$TPU_NAME