import argparse
import torch
import news_sigle_config as config
from model.model_util import ModelConfig
from data_util.data import get_features
import os
import json
import random
import numpy as np
from torch.optim import Adagrad,Adam
from data_util.vocab import Vocab
from model.model import PointerGeneratorNetworks
from train_util import train
from decoder import decoder


def check(args,model_config,vocab):

    train_token_dir = os.path.join(args.token_data,"train")
    test_token_dir = os.path.join(args.token_data,"test")
    val_token_dir = os.path.join(args.token_data,"dev")

    assert os.path.exists(train_token_dir)
    assert os.path.exists(test_token_dir)
    assert os.path.exists(val_token_dir)



    # features_50000_400_100
    feature_dir = "{}_{}_{}_{}".\
        format(args.feature_dir_prefix,model_config.vocab_size,model_config.max_content_len,model_config.max_title_len)
    args.feature_dir = os.path.join(".",feature_dir)

    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
        print("创建的特征目录:{}".format(feature_dir))

    train_feature_dir = os.path.join(".",feature_dir,"train")
    test_feature_dir = os.path.join(".", feature_dir, "test")
    val_feature_dir = os.path.join(".", feature_dir, "val")
    if not os.path.exists(train_feature_dir):
        os.mkdir(train_feature_dir)
    if not os.path.exists(test_feature_dir):
        os.mkdir(test_feature_dir)
    if not os.path.exists(val_feature_dir):
        os.mkdir(val_feature_dir)


    expect_train_feature_file_num = config.expect_train_feature_file_num
    real_train_feature_file_num = len(os.listdir(train_feature_dir))
    if real_train_feature_file_num == 0:
        get_features(token_dir=train_token_dir, feature_dir=train_feature_dir, vocab=vocab, model_config=model_config,
                     data_set="train")
    elif real_train_feature_file_num != expect_train_feature_file_num:
        raise ValueError("train feature dir {} not empty".format(train_feature_dir))

    expect_test_feature_file_num = config.expect_test_feature_file_num
    real_test_feature_file_num = len(os.listdir(test_feature_dir))
    if real_test_feature_file_num == 0:
        get_features(token_dir=test_token_dir, feature_dir=test_feature_dir, vocab=vocab, model_config=model_config,
                     data_set="test")
    elif real_test_feature_file_num != expect_test_feature_file_num:
        raise ValueError("test feature dir {} not empty".format(test_feature_dir))

    expect_dev_feature_file_num = config.expect_dev_feature_file_num
    real_val_feature_file_num = len(os.listdir(val_feature_dir))
    if real_val_feature_file_num == 0:
        get_features(token_dir=val_token_dir, feature_dir=val_feature_dir, vocab=vocab, model_config=model_config,
                     data_set="dev")
    elif real_val_feature_file_num != expect_dev_feature_file_num:
        raise ValueError("val feature dir {} not empty".format(val_feature_dir))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--token_data",default=None,required=True,type = str,
                        help="包含 train，test，dev 和 vocab.json的文件夹")

    parser.add_argument("--feature_dir_prefix",default="features",
                        help="train，test，evl从样本转化成特征所存储的文件夹前缀置")

    parser.add_argument("--do_train",action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_decode", action='store_true',
                        help="是否对测试集进行测试")

    parser.add_argument("--example_num", default= 1024 * 8,type = int,
                        help="每一个特征文件所包含的样本数量")



    parser.add_argument("--no_cuda", action='store_true',
                        help="当GPU可用时，选择不用GPU")

    parser.add_argument("--epoch_num", default=15,type = int,
                        help="epoch")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="train batch size")

    parser.add_argument("--gradient_accumulation_steps", default=4, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="evaluate batch size")


    parser.add_argument("--lr",default=1e-3,type=float,
                        help="learning rate")

    parser.add_argument("--max_grad_norm",default=1.0,type=float,
                        help="Max gradient norm.")

    parser.add_argument("--adagrad_init_acc", default=0.1, type=float,
                        help="adagrad init acc")

    parser.add_argument("--adam_epsilon",default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--output_dir",default="output",type=str,
                        help="Folder to store models and results")

    parser.add_argument("--evaluation_steps",default = 500,type=int,
                        help="Evaluation every N steps of training")
    parser.add_argument("--seed",default=4321,type=int,
                        help="Random seed")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_seed(args.seed)

    vocab_file = os.path.join(args.token_data, 'vocab.json')
    assert os.path.exists(vocab_file)

    model_config_file = os.path.join(".","model","model_config.json")
    assert os.path.exists(model_config_file)
    with open(model_config_file,"r",encoding="utf-8") as f:
        model_config_dict = json.load(f)
        f.close()
    model_config = ModelConfig(**model_config_dict)

    vocab = Vocab(vocab_file=vocab_file, vob_num=model_config.vocab_size)

    model_config.pad_idx = vocab.pad_idx
    model_config.unk_idx = vocab.unk_idx
    model_config.start_idx = vocab.start_idx
    model_config.stop_idx = vocab.stop_idx

    check(args,model_config,vocab)


    model = PointerGeneratorNetworks(config=model_config)


    model = model.to(args.device)
    model = model.to(args.device)
    if args.do_train:
        optimizer = Adam(model.parameters(),lr = args.lr)
        train(args = args,model_config=model_config,model=model,optimizer = optimizer,with_eval = True)
    if args.do_decode:
        decoder(args,model_config=model_config,model=model,vocab=vocab)


if __name__ =="__main__":
    main()