# --original_data_dir
# F:\data\zh\news
# --tokenized_dir
# F:\data\zh\tokenized-single


import code
import argparse
import single_config as config
import os
import json
import random
import collections
import unicodedata



def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split() # 返回一个列表
    return tokens


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False





def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):  # 0xfffd是无法识别的字，数字0 为空，is_control 是否为可打印字符
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)



def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()  # 返回一个列表
    return tokens



def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):

        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)  # 返回的结果是一个使用空格进行split的函数，中文加了空格且返回一个列表
        split_tokens = []
        # 处理流程是先看下是否要变成小写（感觉变成小写就代表了text中有英文），然后使用标点符号把句子拆分
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                # print("\n\n 1 token:",token,"\n\n")
                token = self._run_strip_accents(token)
                # print("\n\n 2 token:",token,"\n\n")

            split_tokens.extend(self._run_split_on_punc(token))

        # 把句中多余的空格去掉，然后返回的是list of token
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 这个函数去除掉text中的非间距字符

        # 标准化对于任何需要以一致的方式处理Unicode文本的程序都是非常重要的。
        # 当处理来自用户输入的字符串而你很难去控制编码的时候尤其如此。
        # normalize() 将文本标准化,第一个参数指定字符串标准化的方式,NFD表示字符应该分解为多个组合字符表示  NFC字符应该是整体组成，可能是使用单一编码 &  NFD字符应该分解为多个组合字符表示
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # category() 返回字符在UNICODE里分类的类型
            cat = unicodedata.category(char)
            if cat == "Mn":
                #  Mark, Nonspacing 指示字符是非间距字符，这指示基字符的修改。
                # https://www.fileformat.info/info/unicode/category/Mn/list.htm
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        # 这个函数使用text中的任意标点符号把句子进行了拆分
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                # 中文的前后增加空格
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 非中文就原样放回了
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class Example(object):
    def __init__(self,original_data_dir,sub_dir,tokenized_output_dir,train_num,test_num,dev_num):

        sub_dir = os.path.join(original_data_dir,sub_dir)
        assert os.path.exists(sub_dir)
        assert 0 != len(os.listdir(sub_dir))

        tokenized_output_dir = os.path.join(tokenized_output_dir)
        if not os.path.exists(tokenized_output_dir):
            os.makedirs(tokenized_output_dir)
        self.train_dir = os.path.join(tokenized_output_dir,"train")
        self.test_dir = os.path.join(tokenized_output_dir,"test")
        self.dev_dir = os.path.join(tokenized_output_dir,"dev")

        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        if not os.path.exists(self.dev_dir):
            os.mkdir(self.dev_dir)

        self.original_data_dir = original_data_dir
        self.sub_dir = sub_dir
        self.tokenized_output_dir = tokenized_output_dir
        self.train_num = train_num
        self.test_num = test_num
        self.dev_num = dev_num



    def read_example(self,tokenizer,sub_file_sample_num):
        total_sample_num = self.train_num + self.test_num + self.dev_num
        train_terminate = self.train_num
        test_terminate = self.train_num + self.test_num
        dev_terminate = self.train_num + self.test_num + self.dev_num


        file_list = os.listdir(self.sub_dir)
        all_sample_list = []
        read_sample_num = 0
        train_file_no = 0
        test_file_no = 0
        dev_file_no = 0
        for file in file_list:
            path_file = os.path.join(self.sub_dir,file)
            with open(path_file,"r",encoding='utf-8') as f:
                data = json.load(f)["data"]
                f.close()
            for sample in data:
                if read_sample_num >= total_sample_num:
                    break
                read_sample_num += 1

                sample_list = []
                news_id = sample['news_id']
                title = sample['title']
                content = sample['content']

                title = " ".join(whitespace_tokenize(clean_text(title))) # 一篇文章的摘要
                content = " ".join(whitespace_tokenize(clean_text(content))) # 摘要对应的正文

                title_token_list = tokenizer.tokenize(title)
                content_token_list = tokenizer.tokenize(content)
                # code.interact(local = locals())

                sample_list.append(int(news_id))
                sample_list.append(title_token_list)
                sample_list.append(content_token_list)
                all_sample_list.append(sample_list)
                if 1 == len(all_sample_list) or len(all_sample_list) % 10000 == 0:
                    print(len(all_sample_list))

                if read_sample_num == train_terminate or \
                        (read_sample_num < train_terminate and len(all_sample_list) == sub_file_sample_num):

                    train_file_no += 1
                    filename = "train_{:0>2d}.json".format(train_file_no)
                    filename = os.path.join(self.train_dir, filename)
                    print("save file {}".format(filename))
                    save_dict = {}
                    save_dict["data"] = all_sample_list
                    with open(filename, 'w', encoding='utf-8') as save_f:
                        json.dump(save_dict, save_f)
                        save_f.close()
                    all_sample_list.clear()
                    save_dict.clear()

                elif (read_sample_num == test_terminate) or \
                        (read_sample_num < test_terminate and len(all_sample_list) == sub_file_sample_num):
                    test_file_no += 1
                    filename = "test_{:0>2d}.json".format(test_file_no)
                    filename = os.path.join(self.test_dir,filename)
                    print("save file {}".format(filename))
                    save_dict = {}
                    save_dict["data"] = all_sample_list

                    with open(filename, 'w', encoding='utf-8') as save_f:
                        json.dump(save_dict, save_f)
                        save_f.close()
                    all_sample_list.clear()
                    save_dict.clear()

                elif (read_sample_num == dev_terminate) or \
                        (read_sample_num < dev_terminate and len(all_sample_list) == sub_file_sample_num):
                    dev_file_no += 1
                    filename = "dev_{:0>2d}.json".format(dev_file_no)
                    filename = os.path.join(self.dev_dir, filename)
                    print("save file {}".format(filename))
                    save_dict = {}
                    save_dict["data"] = all_sample_list
                    with open(filename, 'w', encoding='utf-8') as save_f:
                        json.dump(save_dict, save_f)
                        save_f.close()
                    all_sample_list.clear()
                    save_dict.clear()
                else:
                    pass

            if read_sample_num >= total_sample_num:
                break

    def gene_word_freq(self,vocab_file):
        assert 0 != len(os.listdir(self.train_dir))
        file_list = os.listdir(self.train_dir)
        word_freq = collections.Counter()

        for file in file_list:
            file = os.path.join(self.train_dir,file)
            with open(file,'r',encoding='utf-8') as f:
                data = json.load(f)['data']
                f.close()
            for sample in data:
                title = sample[1]
                content = sample[2]

                word_freq.update(title)
                word_freq.update(content)

        word_freq = word_freq.most_common(len(word_freq))
        word_freq = dict(word_freq)

        vocab_file = os.path.join(self.tokenized_output_dir,vocab_file)
        print("save word freq file {}".format(vocab_file))
        with open(vocab_file,"w",encoding='utf-8') as f:
            json.dump(word_freq,f)
            f.close()


def set_seed(seed):
    random.seed(seed)



def split_original_json(original_json_path,sub_dir,sub_file_sample_num):
    # print("--ffffffff--\n")
    sub_dir = os.path.join(original_json_path,sub_dir)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    # code.interact(local = locals())
    real_sub_file_num = len(os.listdir(sub_dir)) # 0
    except_sub_file_num = config.except_sample_num // sub_file_sample_num # 24
    
    if 0 != config.except_sample_num % sub_file_sample_num:
        except_sub_file_num += 1
    if real_sub_file_num == except_sub_file_num:
        return

    if 0 != real_sub_file_num:
        raise ValueError("子文件夹[{}]不为空，且真实数量与预期不同".format(sub_dir))


    original_json_path = os.path.join(original_json_path, config.file_name)

    sub_file_prefix = "sub"
    sample_num = 0
    file_num = 0
    sub_file_list = []

    # print("asdlfkjj-----\n")
    

    f = open(original_json_path,"r",encoding='utf-8')
    line = "init_str"
    while line:
        line = f.readline().strip()
        
        if "" == line:
            break
        sample_num += 1
        sub_file_list.append(eval(line))
        if sample_num % sub_file_sample_num == 0: # sub_file_sample_num == 100000
            file_num += 1
            current_dict = {}
            assert sub_file_sample_num == len(sub_file_list)
            current_dict["data"] = sub_file_list
            file_name = "{}_{:0>2d}.json".format(sub_file_prefix, file_num)
            file_path_name = os.path.join(sub_dir, file_name)
            # code.interact(local = locals())
            print("第{}-{}个样本，存储于文件{}".format(int(sample_num - sub_file_sample_num + 1), sample_num, file_name))
            with open(file_path_name, "w", encoding='utf-8') as json_file:
                json.dump(current_dict, json_file)
                json_file.close()
            sub_file_list.clear()

    if 0 != len(sub_file_list):
        file_num += 1
        current_dict = {}
        current_dict["data"] = sub_file_list
        file_name = "{}_{:0>2d}.json".format(sub_file_prefix, file_num)
        file_path_name = os.path.join(sub_dir, file_name)
        print("第{}-{}个样本，存储于文件{}".format(int(sample_num - len(sub_file_list) + 1), sample_num, file_name))
        with open(file_path_name, "w", encoding='utf-8') as json_file:
            json.dump(current_dict, json_file)
            json_file.close()
        sub_file_list.clear()
    f.close()





def main():

    parser = argparse.ArgumentParser()

    # F:\data\zh\news
    parser.add_argument("--original_data_dir",default=None,type = str,required=True,
                        help="包含文件news2016zh_train.json的文件夹路径")
    # F:\data\zh\tokenized-single
    parser.add_argument("--tokenized_dir", default=None, type=str, required=True,
                        help="分词后文件所存储的文件夹")

    parser.add_argument("--split_data_dir",default = "E:\\0000_python\\point-genge\\point-generate\\zh-single-word\\news-tokenizer\\sub-single",type=str,
                        help="切分后的文件存储的文件夹(源json文件太大了)")

    parser.add_argument("--sub_file_sample_num",default=1e+5,type=int,
                        help="每个子文件存储的样本数量")
    parser.add_argument("--seed",default=1234,type=int,
                        help="随机种子")


    parser.add_argument("--word_freq",default="vocab.json",type = str,
                        help="词表文件")
    parser.add_argument("--train_sample_num",default=3e+5,type=int,
                        help="训练集样本的数量")
    parser.add_argument("--dev_sample_num",default=1.2e+4,type=int,
                        help="验证集样本的数量")
    parser.add_argument("--test_sample_num",default=1.2e+4,type=int,
                        help="测试集样本的数量")

    args = parser.parse_args()

    set_seed(args.seed)

    split_original_json(original_json_path = args.original_data_dir,
                        sub_dir = args.split_data_dir,
                        sub_file_sample_num = args.sub_file_sample_num)

    # code.interact(local = locals())

    example_obj = Example(original_data_dir = args.original_data_dir,
                          sub_dir = args.split_data_dir,
                          tokenized_output_dir = args.tokenized_dir,
                          train_num = args.train_sample_num,
                          test_num = args.test_sample_num,
                          dev_num = args.test_sample_num,)

    tokenizer = BasicTokenizer()

    # 这里开始没有除以10，内存爆了
    example_obj.read_example(tokenizer,sub_file_sample_num=args.sub_file_sample_num / 10)
    # code.interact(local = locals())

    example_obj.gene_word_freq(args.word_freq)


if __name__ == "__main__":
    main()