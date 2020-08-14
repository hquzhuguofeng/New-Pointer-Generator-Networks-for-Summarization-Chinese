# zn 
指针生成网络，中文数据集下生成摘要, 

# 改动的地方
原论文的指针生成网络，对于正文和摘要的特征抽取是采用单层(双向）的LSTM进行抽取的，我将其变为Bert的embedding。模型的整体框架没有变动，但是工程上的处理进行了微调。

中文数据：
https://github.com/brightmart/nlp_chinese_corpus
250万篇新闻( 原始数据9G，压缩文件3.6G；新闻内容跨度：2014-2016年)
[Google Drive下载](https://drive.google.com/file/d/1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp/view?usp=sharing)或[百度云盘下载](https://pan.baidu.com/share/init?surl=MLLM-CdM6BhJkj8D0u3atA)，密码:k265

### tokenizer
新闻数据集的分词代码


### new-point-generate-zh
指针生成网络在新闻数据集下的应用


## 运行
先是tokenizer
python main.py --original_data_dir E:\0000_python\point-genge\point-generate\zh\data --tokenized_dir ./tokenized_single
E:\0000_python\point-genge\point-generate\zh\datal是我存放新闻数据的地方
这步需要挺多时间的。

然后进入new-point-generate-zh
python main.py --token_data xxx/tokenized --use_coverage --pointer_gen --do_train --do_decode
xxx_toenized 是存放分词后的文件夹

#效果
rouge-1 39%   rouge-2 15% rouge-l 37%









