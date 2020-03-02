import os
import json
import news_sigle_config as config


class Vocab(object):
    def __init__(self,vocab_file,vob_num = 50000):

        # 关于路径，在win下，有时候反斜线会变成转义字符，导致路径不存在
        assert os.path.isfile(vocab_file)
        self.vob_num = vob_num

        self.word_to_idx = {}
        self.idx_to_word = []

        for idx,token in enumerate(config.SPECIAL_TOKEN):
            self.word_to_idx[token] = idx
            self.idx_to_word.append(token)

        self.pad_idx = 0
        self.pad_token = config.SPECIAL_TOKEN[self.pad_idx]

        self.unk_idx = 1
        self.unk_token = config.SPECIAL_TOKEN[self.unk_idx]

        self.start_idx = 2
        self.start_token = config.SPECIAL_TOKEN[self.start_idx]

        self.stop_idx = 3
        self.stop_token = config.SPECIAL_TOKEN[self.stop_idx]

        with open(vocab_file,"r",encoding='utf-8') as f:
            word_freq = json.load(f)
        f.close()

        special_len = len(self.idx_to_word)

        for i,token in enumerate(word_freq):
            idx = special_len + i
            if idx >= self.vob_num:
                break
            self.word_to_idx[token] = idx
            self.idx_to_word.append(token)


    def word_2_idx(self,word):
        return self.word_to_idx.get(word,self.unk_idx)



    def idx_2_word(self,idx):
        if (idx >= 0) and (idx < self.vob_num):
            return self.idx_to_word[idx]
        else:
            return self.unk_idx


    def get_vob_size(self):
        return self.vob_num













