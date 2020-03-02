import os
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from rouge import Rouge
import json




def get_features_from_cache(cache_file):

    features = torch.load(cache_file)

    all_encoder_input = torch.tensor([f.encoder_input for f in features], dtype=torch.long)
    all_encoder_mask = torch.tensor([f.encoder_mask for f in features],dtype = torch.long)

    all_decoder_input = torch.tensor([f.decoder_input for f in features],dtype=torch.long)
    all_decoder_mask = torch.tensor([f.decoder_mask for f in features],dtype=torch.int)

    all_decoder_target = torch.tensor([f.decoder_target for f in features],dtype=torch.long)

    all_encoder_input_with_oov = torch.tensor([f.encoder_input_with_oov for f in features],dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor([f.decoder_target_with_oov for f in features],dtype=torch.long )
    all_oov_len = torch.tensor([f.oov_len for f in features],dtype=torch.int)

    dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                            all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len)

    news_ids = [f.news_id for f in features]
    titles = [f.title for f in features]
    oovs = [f.oovs for f in features]
    return dataset,news_ids,oovs,titles


def from_feature_get_model_input(features,hidden_dim,device = torch.device("cpu"),pointer_gen = True,
                                 use_coverage = True):

    all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
    all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len = features


    max_encoder_len = all_encoder_mask.sum(dim=-1).max()
    max_decoder_len = all_decoder_mask.sum(dim=-1).max()

    all_encoder_input = all_encoder_input[:,:max_encoder_len]
    all_encoder_mask = all_encoder_mask[:,:max_encoder_len]

    all_encoder_input_with_oov = all_encoder_input_with_oov[:,:max_encoder_len]
    all_decoder_target_with_oov = all_decoder_target_with_oov[:,:max_decoder_len]



    batch_size = all_encoder_input.shape[0]
    max_oov_len = all_oov_len.max().item()

    oov_zeros = None
    if pointer_gen:                # 当时用指针网络时，decoder_target应该要带上oovs
        all_decoder_target = all_decoder_target_with_oov
        if max_oov_len > 0:                # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
            oov_zeros = torch.zeros((batch_size, max_oov_len),dtype= torch.float32)
    else:                                  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
        all_encoder_input_with_oov = None


    init_coverage = None
    if use_coverage:
        init_coverage = torch.zeros(all_encoder_input.size(),dtype=torch.float32)          # 注意数据格式是float

    init_context_vec = torch.zeros((batch_size, 2 * hidden_dim),dtype=torch.float32)   # 注意数据格式是float

    model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,
                   init_coverage]
    model_input = [t.to(device) if t is not None else None for t in model_input]

    return model_input


def idx_to_token(idx,oov_word,vocab):

    if idx < vocab.vob_num:
        return vocab.idx_2_word(idx)
    else:
        idx = idx - vocab.vob_num

        if idx < len(oov_word):
            return oov_word[idx]
        else:
            return vocab.unk_token


def get_evaluate_top_k(output_dir,k = 5):
    step_dir_list = os.listdir(output_dir)
    step_dir_list = [os.path.join(output_dir, step_dir) for step_dir in step_dir_list]
    step_dir_list = [step_dir for step_dir in step_dir_list if os.path.isdir(step_dir)]

    all_step = []
    all_eval = []
    for step_dir in step_dir_list:
        step_dir_path = os.path.join(step_dir)

        val_loss_file = os.path.join(step_dir_path, "eval_loss.json")
        step_num = int(step_dir.split("_")[-1])
        all_step.append(step_num)

        with open(val_loss_file, "r", encoding='utf-8') as f:
            eval_loss_list = dict(json.load(f))
            f.close()


        eval_loss_list = [v for k, v in eval_loss_list.items()]


        eval_loss = sum(eval_loss_list) / len(eval_loss_list)

        all_eval.append(eval_loss)

    all_step_loss = dict(zip(all_step, all_eval))
    all_step_loss_sorted = list(dict(sorted(all_step_loss.items(), key=lambda x: x[1])).keys())
    topk_step_list = all_step_loss_sorted[:k]
    topk_step_list = [str(step) for step in topk_step_list]
    topk_dir = []
    for step_dir in step_dir_list:
        if any([step in step_dir for step in topk_step_list]):
            topk_dir.append(step_dir)
    return topk_dir



def decoder(args,model_config,model,vocab):
    # output_dir = args.output_dir
    # model_dir_list = os.listdir(output_dir)
    # model_dir_list = [os.path.join(".",output_dir,model_dir) for model_dir in model_dir_list]
    # model_dir_list = [model_dir for model_dir in model_dir_list if os.path.isdir(model_dir)]


    model_dir_list = get_evaluate_top_k(args.output_dir)

    decoder_info_str = "\n".join(model_dir_list)
    decoder_info_file = os.path.join(args.output_dir,"decoder.txt")
    with open(decoder_info_file,"w",encoding='utf-8') as f:
        f.write(decoder_info_str)
    print("将会对以下几个输出文件夹编码\n{}".format(decoder_info_str))

    test_feature_dir = os.path.join(args.feature_dir, "test")
    feature_file_list = os.listdir(test_feature_dir)

    rouge = Rouge()

    model_iterator = trange(int(len(model_dir_list)), desc = "Model.bin File")
    with torch.no_grad():
        for model_idx in model_iterator:
            model_dir = model_dir_list[model_idx]

            decoder_dir = model_dir
            predict_file = os.path.join(decoder_dir,"predict.txt")
            score_json = {}
            score_json_file = os.path.join(decoder_dir,"score.json")
            result_json = {}
            result_json_file = os.path.join(decoder_dir,"result.json")

            model_path_name = os.path.join(model_dir,"model.bin")
            model.load_state_dict(torch.load(model_path_name))
            model = model.to(args.device)
            model.eval()

            file_iterator = trange(int(len(feature_file_list)), desc=decoder_dir)

            for file_idx in file_iterator:
                file = feature_file_list[file_idx]
                path_file = os.path.join(test_feature_dir,file)

                test_dataset,news_ids,oovs,titles = get_features_from_cache(path_file)
                test_sampler = SequentialSampler(test_dataset)
                train_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

                data_iterator = tqdm(train_dataloader, desc=decoder_dir)
                for i, batch in enumerate(data_iterator):
                    batch = from_feature_get_model_input(batch, hidden_dim=model_config.hidden_dim, device=args.device,
                                                         pointer_gen=model_config.pointer_gen,
                                                         use_coverage=model_config.use_coverage)
                    news_id = news_ids[i]
                    current_oovs = oovs[i]
                    current_title = titles[i][1:-1]        # 去掉start,stop
                    beam = model(encoder_input = batch[0],
                                 encoder_mask= batch[1],
                                 encoder_with_oov = batch[2],
                                 oovs_zero = batch[3],
                                 context_vec = batch[4],
                                 coverage = batch[5],
                                 mode = "decode",
                                 beam_size = 10
                                 )
                    # 去除 start token
                    hypothesis_idx_list = beam.tokens[1:]
                    if vocab.stop_idx == hypothesis_idx_list[-1]:
                        hypothesis_idx_list = hypothesis_idx_list[:-1]


                    hypothesis_token_list = [idx_to_token(index,oov_word = current_oovs,vocab = vocab)
                                             for index in hypothesis_idx_list]

                    hypothesis_str = " ".join(hypothesis_token_list)
                    reference_str = " ".join(current_title)

                    result_str = "{}\t{}\t{}\n".format(news_id,reference_str,hypothesis_str)
                    with open(file=predict_file,mode='a',encoding='utf-8') as f:
                        f.write(result_str)
                        f.close()
                    rouge_score = rouge.get_scores(hyps = hypothesis_str,refs= reference_str)
                    score_json[news_id] = rouge_score[0]

            with open(score_json_file, 'w') as f:
                json.dump(score_json,f)
                f.close()



            rouge_1_f = []
            rouge_1_p = []
            rouge_1_r = []
            rouge_2_f = []
            rouge_2_p = []
            rouge_2_r = []
            rouge_l_f = []
            rouge_l_p = []
            rouge_l_r = []


            for name,score in score_json.items():
                rouge_1_f.append(score["rouge-1"]['f'])
                rouge_1_p.append(score["rouge-1"]['p'])
                rouge_1_r.append(score["rouge-1"]['r'])
                rouge_2_f.append(score["rouge-2"]['f'])
                rouge_2_p.append(score["rouge-2"]['p'])
                rouge_2_r.append(score["rouge-2"]['r'])
                rouge_l_f.append(score["rouge-l"]['f'])
                rouge_l_p.append(score["rouge-l"]['p'])
                rouge_l_r.append(score["rouge-l"]['r'])

            mean_1_f = sum(rouge_1_f) / len(rouge_1_f)
            mean_1_p = sum(rouge_1_p) / len(rouge_1_p)
            mean_1_r = sum(rouge_1_r) / len(rouge_1_r)
            mean_2_f = sum(rouge_2_f) / len(rouge_2_f)
            mean_2_p = sum(rouge_2_p) / len(rouge_2_p)
            mean_2_r = sum(rouge_2_r) / len(rouge_2_r)
            mean_l_f = sum(rouge_l_f) / len(rouge_l_f)
            mean_l_p = sum(rouge_l_p) / len(rouge_l_p)
            mean_l_r = sum(rouge_l_r) / len(rouge_l_r)



            result_json['mean_1_f'] = mean_1_f
            result_json['mean_1_p'] = mean_1_p
            result_json['mean_1_r'] = mean_1_r
            result_json['mean_2_f'] = mean_2_f
            result_json['mean_2_p'] = mean_2_p
            result_json['mean_2_r'] = mean_2_r
            result_json['mean_l_f'] = mean_l_f
            result_json['mean_l_p'] = mean_l_p
            result_json['mean_l_r'] = mean_l_r
            with open(result_json_file, 'w') as f:  # test.json文本，只能写入状态 如果没有就创建
                json.dump(result_json, f)  # data转换为json数据格式并写入文件
                f.close()



