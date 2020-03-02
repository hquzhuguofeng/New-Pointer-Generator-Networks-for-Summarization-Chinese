
import os
import torch
import logging
from random import shuffle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import json
import news_sigle_config as config


# self.encoder_input, self.encoder_mask = self._add_pad_and_gene_mask(encoder_input, max_encoder_len, pad_idx)
#
# self.decoder_input, self.decoder_mask = self._add_pad_and_gene_mask(decoder_input, max_decoder_len, pad_idx)
#
# self.decoder_target = decoder_target
#
# self.encoder_input_with_oov = encoder_input_with_oov
# self.oovs = oovs
#
# self.decoder_target_with_oov = decoder_target_with_oov


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



    return dataset


def from_feature_get_model_input(features,hidden_dim,device = torch.device("cpu"),pointer_gen = True,
                                 use_coverage = True):

    all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
    all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len = features


    max_encoder_len = all_encoder_mask.sum(dim=-1).max()
    max_decoder_len = all_decoder_mask.sum(dim=-1).max()

    all_encoder_input = all_encoder_input[:,:max_encoder_len]
    all_encoder_mask = all_encoder_mask[:,:max_encoder_len]
    all_decoder_input = all_decoder_input[:,:max_decoder_len]
    all_decoder_mask = all_decoder_mask[:,:max_decoder_len]
    all_decoder_target = all_decoder_target[:,:max_decoder_len]
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
                   init_coverage,all_decoder_input,all_decoder_mask,all_decoder_target]
    model_input = [t.to(device) if t is not None else None for t in model_input]

    return model_input


def evaluate(args,model_config,model,save_dir,update_step):
    eval_feature_dir = os.path.join(args.feature_dir, "val")
    feature_file_list = os.listdir(eval_feature_dir)

    batch_size = args.eval_batch_size
    hidden_dim = model_config.hidden_dim
    device = args.device
    pointer_gen = model_config.pointer_gen
    use_coverage = model_config.use_coverage
    total_loss = 0.
    sample_num = 0.

    model.eval()
    shuffle(feature_file_list)
    file_iterator = trange(int(len(feature_file_list)), desc="Evaluate File")
    with torch.no_grad():
        for file_idx in file_iterator:
            file_name = feature_file_list[file_idx]
            file_path_name = os.path.join(eval_feature_dir, file_name)
            assert os.path.isfile(file_path_name)
            train_dataset = get_features_from_cache(file_path_name)

            eval_sampler = SequentialSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=batch_size)
            data_iterator = tqdm(train_dataloader, desc="Evaluate Iteration")
            for i, batch in enumerate(data_iterator):
                batch = from_feature_get_model_input(batch, hidden_dim=hidden_dim, device=device,
                                                     pointer_gen=pointer_gen, use_coverage=use_coverage)

                inputs = {'encoder_input': batch[0],
                          'encoder_mask': batch[1],
                          'encoder_with_oov': batch[2],
                          'oovs_zero': batch[3],
                          'context_vec': batch[4],
                          'coverage': batch[5],
                          'decoder_input': batch[6],
                          'decoder_mask': batch[7],
                          'decoder_target': batch[8],
                          'mode':'eval'}

                loss = model(**inputs)

                true_batch_size = batch[0].shape[0]
                sample_num += true_batch_size
                batch_loss = loss * true_batch_size
                total_loss += batch_loss


        avg_loss = (total_loss / float(sample_num)).item()
        logging.info("evaluate on step = [{}] , average loss : [{}]".format(update_step, avg_loss))
        avg_loss_dict = {update_step: avg_loss}
        avg_loss_path_file = os.path.join(save_dir,"eval_loss.json")

        with open(avg_loss_path_file, 'w',encoding='utf-8') as f:
            json.dump(avg_loss_dict, f)
            f.close()
        logging.info("update step:[{}] ,save Evaluate loss file[{}]".
                     format(update_step, avg_loss_path_file))

    model.train()
    # over























def train_check(output_dir):
    output_dir = os.path.join(".", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if 0 != len(os.listdir(output_dir)):
        raise ValueError("output dir not empty.")
    logfile = os.path.join(".", output_dir, "logging.log")
    # 如果没有参数logfile，信息就打印在控制台上
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=logfile,
                        level=logging.INFO)



def train(args,model_config,model,optimizer,with_eval):


    train_check(output_dir = args.output_dir)


    train_feature_dir = os.path.join(args.feature_dir, "train")
    feature_file_list = os.listdir(train_feature_dir)

    epoch_num = args.epoch_num
    batch_size = args.train_batch_size
    hidden_dim = model_config.hidden_dim
    device = args.device
    pointer_gen = model_config.pointer_gen
    use_coverage = model_config.use_coverage

    update_step = 0
    forward_step = 0
    current_loss = 0.0
    step_loss = {}

    model.train()
    model.zero_grad()
    epoch_iterator = trange(int(epoch_num), desc="Epoch")

    logging.info("***** Running training *****")
    logging.info("  Num Epochs = {}".format(epoch_num))
    logging.info("  sample num = {}".format(config.expect_train_sample_num))
    logging.info("  batch_size = {}".format(batch_size))
    logging.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))


    for epoch in epoch_iterator:
        shuffle(feature_file_list)
        file_iterator = trange(int(len(feature_file_list)), desc="Train File")
        for file_idx in file_iterator:
            file_name = feature_file_list[file_idx]
            file_path_name = os.path.join(train_feature_dir,file_name)
            assert os.path.isfile(file_path_name)
            train_dataset = get_features_from_cache(file_path_name)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
            data_iterator = tqdm(train_dataloader, desc="Train Iteration")
            for i,batch in enumerate(data_iterator):

                batch = from_feature_get_model_input(batch,hidden_dim=hidden_dim,device = device,
                                                     pointer_gen = pointer_gen,use_coverage = use_coverage)

                inputs = {'encoder_input':batch[0],
                          'encoder_mask':batch[1],
                          'encoder_with_oov':batch[2],
                          'oovs_zero':batch[3],
                          'context_vec':batch[4],
                          'coverage':batch[5],
                          'decoder_input':batch[6],
                          'decoder_mask':batch[7],
                          'decoder_target':batch[8]}
                model.train()
                loss = model(**inputs)

                forward_step += 1
                if args.gradient_accumulation_steps > 1:
                    loss /= float(args.gradient_accumulation_steps)
                loss.backward()
                current_loss += loss.item()
                if 0 == (forward_step % args.gradient_accumulation_steps):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()

                    update_step += 1

                    logging.info("EPOCH : [{}/{}] File:[{}] update_step:{} loss {:.2f}".
                                 format(epoch+1,epoch_num,file_name,update_step,current_loss))

                    step_loss[update_step] = current_loss
                    current_loss = 0.0

                    if 1 == update_step or 0 == update_step % args.evaluation_steps:

                        save_dir = os.path.join(args.output_dir, "step_{:05d}".format(update_step))
                        os.mkdir(save_dir)
                        torch.save(model.state_dict(), os.path.join(save_dir, "model.bin"))
                        logging.info("update step:[{}] ,save model dir[{}]".format(update_step,
                                                                                   os.path.join(save_dir, "model.bin")))
                        avg_loss_path_file = os.path.join(save_dir, "train_loss.json")
                        with open(avg_loss_path_file, 'w', encoding='utf-8') as f:
                            json.dump(step_loss, f)
                            f.close()
                        step_loss = {}
                        logging.info("update step:[{}] ,save Train loss file[{}]".
                                     format(update_step, avg_loss_path_file))

                        if with_eval:
                            evaluate(args=args,model_config=model_config,model=model,save_dir=save_dir,update_step=update_step)


                        else:
                            pass
