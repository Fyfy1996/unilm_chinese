# -*- coding: utf-8 -*-


# evaluate


from modeling_unilm import UnilmForSeq2SeqDecode
from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
from modeling_unilm import UnilmForSeq2Seq, UnilmConfig
import torch
import os
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2Seq, UnilmTokenizer)
}
config_class, model_class, tokenizer_class = MODEL_CLASSES["unilm"]
model_recover = torch.load(os.path.join("output_dir_1", "model.3.bin"), map_location='cpu')
model = UnilmForSeq2SeqDecode.from_pretrained("unilm_chinese", state_dict=model_recover) 



import torch
import utils_seq2seq
from transformers import BertTokenizer
from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
tokenizer = UnilmTokenizer.from_pretrained("unilm_chinese", do_lower_case=True)
data_tokenizer = tokenizer


val_pipeline = [utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), 
                                                       tokenizer.convert_tokens_to_ids)]



train_dataset = utils_seq2seq.Seq2SeqDataset("data/csl_title_public/csl_title_train.json", 4, 
     data_tokenizer, 512, bi_uni_pipeline=val_pipeline)


import torch
from torch.utils.data import RandomSampler
train_sampler = RandomSampler(train_dataset, replacement=False)
_batch_size = 2

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, 
                                               sampler=train_sampler,
                                               num_workers=2, 
                                               collate_fn=utils_seq2seq.batch_list_to_batch_tensors, 
                                               pin_memory=False)

text,true_re = train_dataset.ex_list[89]
input_ids, segment_ids, position_ids, input_mask = val_pipeline[0](([ i for i in text], 400))

input_tensor = torch.LongTensor([input_ids])
seg_tensor = torch.LongTensor([segment_ids])
pos_tensor = torch.LongTensor([position_ids])
mask_tensor = input_mask.unsqueeze(0)

preds = model(input_tensor, seg_tensor, pos_tensor, mask_tensor)

re = ""
str_list = tokenizer.convert_ids_to_tokens(preds[0].tolist())
for i in str_list:
    if i == '[SEP]':
        break
    re += i
    
print(re)
print(true_re)

print(text)





