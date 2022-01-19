nohup python -u run_seq2seq.py --data_dir data/CCPC --src_file ccpc_train_v1.0.json --model_type unilm --model_name_or_path unilm_chinese --output_dir poet_bot --max_seq_length 512 --max_position_embeddings 512 --do_train --do_lower_case --train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 3 > log_poem.log 2>&1 &