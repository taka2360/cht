from transformers import BertJapaneseTokenizer, BertModel
import torch
import torch.nn.functional as F
import pandas as pd



class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    

print("ベクトル変換用のmodelを生成しています...")
model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")
print("modelを生成しました")

import tensorflow as tf
import tensorflow_datasets as tfds

builder = tfds.builder('huggingface:cc100/lang=ja')
builder.download_and_prepare()
ds = builder.as_dataset(split='train', shuffle_files=False)

for i, x in enumerate(ds.take(100)):
    text_tensor = x['text']
    text_str = tf.compat.as_text(text_tensor.numpy())
    print(f'{i} : {text_str}')

"""
input_docs = [
    'あなたは犬が',
    '好き',
]
vecs = model.encode(input_docs, batch_size=12)
print(vecs)"""