import os
import json
import torch
import news_sigle_config as config


class Feature(object):
    def __init__(self, news_id, title, content, encoder_input, decoder_input, decoder_target,
                 encoder_input_with_oov, oovs,decoder_target_with_oov, max_encoder_len, max_decoder_len, pad_idx=0):

        assert len(decoder_input) == len(decoder_target)
        self.news_id = news_id    # int
        self.title = title      # str
        self.content = content  # str

        self.encoder_input, self.encoder_mask = self._add_pad_and_gene_mask(encoder_input, max_encoder_len, pad_idx)
        self.encoder_input_with_oov = self._add_pad_and_gene_mask(encoder_input_with_oov, max_encoder_len, pad_idx,
                                                                  return_mask=False)
        self.decoder_input, self.decoder_mask = self._add_pad_and_gene_mask(decoder_input, max_decoder_len, pad_idx)

        self.decoder_target = self._add_pad_and_gene_mask(decoder_target, max_decoder_len, pad_idx, return_mask=False)
        self.decoder_target_with_oov = self._add_pad_and_gene_mask(decoder_target_with_oov, max_decoder_len, pad_idx,
                                                                   return_mask=False)
        self.oovs = oovs
        self.oov_len = len(oovs)


    @classmethod
    def _add_pad_and_gene_mask(cls,x,max_len,pad_idx = 0,return_mask = True):
        pad_len = max_len - len(x)
        assert pad_len >= 0

        if return_mask:
            mask = [1] * len(x)
            mask.extend([0] * pad_len)
            assert len(mask) == max_len

        x.extend([pad_idx] * pad_len)
        assert len(x) == max_len

        if return_mask:
            return x,mask
        else:
            return x



def content_word_to_idx_with_oov(content_list,vocab):
    indexes = []
    oovs = []
    for word in content_list:
        idx = vocab.word_2_idx(word)
        if vocab.unk_idx == idx:
            if word not in oovs:
                oovs.append(word)
            oov_idx = oovs.index(word)
            indexes.append(vocab.get_vob_size() + oov_idx)
        else:
            indexes.append(idx)
    return indexes,oovs


def target_target_idx_with_oov(title_list, vocab , oovs):
    target_with_oov = []
    for word in title_list[1:]:
        idx = vocab.word_2_idx(word)
        if vocab.unk_idx == idx:
            if word in oovs:
                target_with_oov.append(vocab.vob_num+oovs.index(word))
            else:
                target_with_oov.append(vocab.unk_idx)
        else:
            target_with_oov.append(idx)
    return target_with_oov







def from_sample_covert_feature(vocab,news_id,title,content,index,content_len,title_len,point = True):

    if 0 == len(title) or 0 == len(content):
        return None
    if len(content) <= len(title):
        return None

    print_idx = 20
    if index < print_idx:
        print("====================================={}=====================================".format(news_id))

    if index < print_idx:
        print("原始内容长度[{}]===[{}]".format(len(content), " ".join(content)))
        print("原始标题长度[{}]===[{}]".format(len(title), " ".join(title)))

    content = content[:content_len]

    if index < print_idx:
        print("截断后的内容长度[{}]===[{}]".format(len(content)," ".join(content)))

    encoder_input = [vocab.word_2_idx(word) for word in content]

    # 加上 start 和 end
    title = [vocab.start_token] + title + [vocab.stop_token]
    # 截断，限制摘要的长度
    title = title[:title_len + 1]

    if index < print_idx:
        print("截断后的标题长度[{}]===[{}]".format(len(title)," ".join(title)))

    title_indexes = [vocab.word_2_idx(word) for word in title]

    decoder_input = title_indexes[:-1]
    decoder_target = title_indexes[1:]

    assert len(decoder_input) == len(decoder_target)

    if point:
        encoder_input_with_oov,oovs = content_word_to_idx_with_oov(content,vocab)
        decoder_target_with_oov = target_target_idx_with_oov(title,vocab,oovs)

    feature_obj = Feature(news_id=news_id,
                          title=title,
                          content=content,
                          encoder_input=encoder_input,
                          decoder_input=decoder_input,
                          decoder_target=decoder_target,
                          encoder_input_with_oov=encoder_input_with_oov,
                          oovs=oovs,
                          decoder_target_with_oov=decoder_target_with_oov,
                          max_encoder_len=content_len,
                          max_decoder_len=title_len,
                          pad_idx=vocab.pad_idx)

    if index < print_idx:
        print("encoder_input :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_input])))
        print("encoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_mask])))
        print("encoder_input_with_oov :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_input_with_oov])))
        print("decoder_input :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_input])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_mask])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_target])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_target_with_oov])))
        print("oovs          :[{}]".format(" ".join(oovs)))
        print("\n")


    return feature_obj






def get_features(token_dir,feature_dir,vocab,model_config,data_set = "train",example_num = 1024*8):
    assert os.path.exists(token_dir)
    assert os.path.exists(feature_dir)
    assert data_set in ["train","test","dev"]
    assert 0 == example_num % 1024 and example_num > 1024

    if "train" == data_set:
        sample_num = config.expect_train_sample_num
    elif "test" == data_set:
        sample_num = config.expect_test_sample_num
    elif "dev" == data_set:
        sample_num = config.expect_dev_sample_num
    else:
        pass

    feature_file_prefix = "{}".format(data_set)
    features = []

    token_file_list = os.listdir(token_dir)
    index = 0
    feature_file_idx = 0
    for file in token_file_list:
        file = os.path.join(token_dir,file)
        with open(file,'r',encoding='utf-8') as f:
            data = json.load(f)['data']
            f.close()
        for sample in data:

            if example_num == len(features):
                feature_file_idx += 1
                feature_file_name = "{}_{:0>2d}".format(feature_file_prefix, feature_file_idx)
                feature_file_name_path = os.path.join(feature_dir, feature_file_name)
                torch.save(features, feature_file_name_path)
                print("本次转换完成{},已经转换完成{}个，一共{}个,占比{:.2%}  存储特征文件{}".
                      format(len(features), index, sample_num, float(index) / sample_num, feature_file_name))
                features = []
            index += 1
            news_id = sample[0]
            title = sample[1]
            content = sample[2]

            feature = from_sample_covert_feature(vocab = vocab,
                                                 news_id = news_id,
                                                 title = title,
                                                 content = content,
                                                 index = index,
                                                 content_len = model_config.max_content_len,
                                                 title_len = model_config.max_title_len,
                                                 point = model_config.pointer_gen)
            if feature is not None:
                features.append(feature)

    if 0 != len(features):
        feature_file_idx += 1
        feature_file_name = "{}_{:0>2d}".format(feature_file_prefix, feature_file_idx)
        feature_file_name_path = os.path.join(feature_dir, feature_file_name)
        torch.save(features, feature_file_name_path)

        print("本次转换完成{},已经转换完成{}个，一共{}个,占比{:.2%}  存储特征文件{}".
              format(len(features), index, sample_num, float(index + 1) / sample_num, feature_file_name))
        features = []


