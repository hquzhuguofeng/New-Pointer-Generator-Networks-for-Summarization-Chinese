import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
import math
from model.model_util import Beam,sort_beams,ModelConfig





BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = gelu



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_idx)
        self.position_embeddings = nn.Embedding(config.max_content_len, config.hidden_dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_dim, eps=config.eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids,position_ids = None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.LayerNorm = BertLayerNorm(config.hidden_dim, eps=config.eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs, input_tensor)
        outputs = attention_output   # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.intermediate_act_fn = ACT2FN



    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.LayerNorm = BertLayerNorm(config.hidden_dim, eps=config.eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):


        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output







class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()

        self.config = config

        self.embedding = BertEmbeddings(config)
        self.bertLayer = BertLayer(config)

        self.lstm = nn.LSTM(input_size = config.embedding_dim,
                            hidden_size = config.hidden_dim,
                            num_layers = config.encoder_lstm_num_layer,
                            dropout = config.hidden_dropout_prob,
                            batch_first = True,
                            bidirectional = True)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)


    # x.shape (batch,seq_len)  词的索引
    # mask.shape (batch,seq_len)   每个样本的真实长度
    def forward(self,x,mask):

        embedded = self.dropout(self.embedding(x)) # 这里的dropout可以不用加了

        bert_output = self.bertLayer(hidden_states = embedded,attention_mask = mask)
        bool_mask = (mask == 0)
        # 在mask的位置用value来填充，
        bert_output_masked = bert_output.masked_fill(mask=bool_mask.unsqueeze(dim = 2),value=0)

        seq_lens = mask.sum(dim = -1) # 统计每个句子的长度，【batch】

        packed = pack_padded_sequence(input=bert_output_masked,lengths=seq_lens,batch_first=True,enforce_sorted=False)
        output_packed,(h,c) = self.lstm(packed)

        # pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):

        output,_ = pad_packed_sequence(sequence = output_packed,
                                       batch_first = True,
                                       padding_value=self.config.pad_idx,
                                       total_length = seq_lens.max())

        return output,(h,c)



class Reduce(nn.Module):
    def __init__(self,config):
        super(Reduce,self).__init__()

        self.config = config

        self.reduce_h = nn.Linear(config.hidden_dim*2,config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim*2,config.hidden_dim)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self,h,c):


        assert self.config.hidden_dim == h.shape[2]
        assert self.config.hidden_dim == c.shape[2]

        h = h.reshape(-1,self.config.hidden_dim*2)
        c = c.reshape(-1,self.config.hidden_dim*2)

        h_output = self.dropout(self.reduce_h(h))
        c_output = self.dropout(self.reduce_c(c))

        h_output = F.relu(h_output)
        c_output = F.relu(c_output)

        # h_output.shape = c_output.shape =
        # (batch,hidden)
        return h_output.unsqueeze(0),c_output.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self,config):

        # hidden_dim, use_coverage = False
        super(Attention, self).__init__()

        self.config = config

        self.w_s = nn.Linear(config.hidden_dim*2,config.hidden_dim*2,bias=False)

        if config.use_coverage:
            self.w_c = nn.Linear(1,config.hidden_dim*2)

        self.v = nn.Linear(config.hidden_dim*2,1,bias=False)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    # h ：encoder hidden states h_i ,On each step t. (batch,seq_len,hidden*2)
    # mask, 0-1 encoder_mask (batch,seq_len)
    # s : decoder state s_t,one step (batch,hidden*2)
    # coverage : sum of attention score (batch,seq_len)
    def forward(self,encoder_features,mask,s,coverage):

        decoder_features = self.dropout(self.w_s(s).unsqueeze(1))  # (batch,1,hidden*2)

        # broadcast 广播运算
        attention_feature = encoder_features + decoder_features  # (batch,seq_len,hidden*2)


        if self.config.use_coverage:
            coverage_feature = self.dropout(self.w_c(coverage.unsqueeze(2))) # (batch,seq_len,hidden*2)
            attention_feature += coverage_feature


        e_t = self.dropout(self.v(torch.tanh(attention_feature)).squeeze(dim = 2))   # (batch,seq_len)

        mask_bool = (mask == 0)   # mask pad position eq True

        e_t.masked_fill_(mask = mask_bool,value= -float('inf'))

        a_t = torch.softmax(e_t,dim=-1)  # (batch,seq_len)

        if self.config.use_coverage:
            next_coverage = coverage + a_t

        return a_t,next_coverage


class GeneraProb(nn.Module):
    def __init__(self,config):
        super(GeneraProb,self).__init__()

        self.w_h = nn.Linear(config.hidden_dim*2,1)
        self.w_s = nn.Linear(config.hidden_dim*2,1)
        self.w_x = nn.Linear(config.embedding_dim,1)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    # h : weight sum of encoder output ,(batch,hidden*2)
    # s : decoder state                 (batch,hidden*2)
    # x : decoder input                 (batch,embed)
    def forward(self,h,s,x):
        h_feature = self.dropout(self.w_h(h))     # (batch,1)
        s_feature = self.dropout(self.w_s(s))   # (batch,1)
        x_feature = self.dropout(self.w_x(x))    # (batch,1)

        gen_feature = h_feature + s_feature + x_feature  # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p


class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()

        self.config = config


        self.bertLayer = BertLayer(config)

        self.get_lstm_input = nn.Linear(in_features=config.hidden_dim * 2 + config.embedding_dim,
                                        out_features=config.embedding_dim)

        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.decoder_lstm_num_layer,
                            dropout=config.hidden_dropout_prob,
                            batch_first=True,
                            bidirectional=False)

        self.attention = Attention(config)

        if config.pointer_gen:
            self.genera_prob = GeneraProb(config)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)


        # self.out = nn.Sequential(nn.Linear(in_features=hidden_dim * 3,out_features=embed_dim),
        #                          self.dropout(),
        #                          nn.ReLU(),
        #                          nn.Linear(in_features=hidden_dim,out_features=vob_size),
        #                          self.dropout)
        self.out = nn.Sequential(nn.Linear(in_features=config.hidden_dim * 3, out_features=config.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=config.hidden_dim, out_features=config.vocab_size))

    # decoder_input_one_step (batch,t)   (t = step +1)
    # decoder_status = (h_t,c_t)  h_t (1,batch,hidden)
    # encoder_output (batch,seq_len,hidden*2)
    # encoder_mask (batch,seq_len)
    # context_vec (bach,hidden*2)  encoder weight sum about attention score
    # oovs_zero (batch,max_oov_size)  all-zero tensor
    # encoder_with_oov (batch,seq_len)  Index of words in encoder_with_oov can be greater than vob_size
    # coverage : Sum of attention at each step
    # step...
    def forward(self,decoder_embed_one_step,decoder_mask_one_step,decoder_status,encoder_output,encoder_features,
                encoder_mask,context_vec,oovs_zero,encoder_with_oov,coverage,step):



        bert_output = self.bertLayer(hidden_states=decoder_embed_one_step,attention_mask = decoder_mask_one_step)
        bool_mask = (decoder_mask_one_step == 0)
        bert_output_masked = bert_output.masked_fill(mask=bool_mask.unsqueeze(dim=2), value=0)
        bert_output_masked_last_step = bert_output_masked[:,-1,:]
        x = self.get_lstm_input(torch.cat([context_vec, bert_output_masked_last_step], dim=-1)).unsqueeze(dim=1)  # (batch,1,hidden*2+embed_dim)


        decoder_output,next_decoder_status = self.lstm(x,decoder_status)

        h_t,c_t = next_decoder_status

        batch_size = c_t.shape[1]

        h_t_reshape = h_t.reshape(batch_size,-1)
        c_t_reshape = c_t.reshape(batch_size,-1)

        status = torch.cat([h_t_reshape,c_t_reshape],dim = -1)   # (batch,hidden_dim*2)

        # attention_score (batch,seq_len)  Weight of each word vector
        # next_coverage (batch,seq_len)  sum of attention_score


        attention_score,next_coverage = self.attention(encoder_features = encoder_features,
                                                       mask = encoder_mask,
                                                       s = status,
                                                       coverage = coverage)




        # (batch,hidden_dim*2)  encoder_output weight sum about attention_score
        # current_context_vec = torch.bmm(attention_score.unsqueeze(1),encoder_output).squeeze()
        # 尝试一下高级写法，结果和上面一行一致
        current_context_vec = torch.einsum("ab,abc->ac",attention_score,encoder_output)

        # (batch,1)
        genera_p = None
        if self.config.pointer_gen:
            genera_p = self.genera_prob(h = current_context_vec,
                                        s = status,
                                        x = x.squeeze())

        # (batch,hidden_dim*3)
        out_feature = torch.cat([decoder_output.squeeze(dim = 1),current_context_vec],dim = -1)

        # (batch,vob_size)
        output = self.out(out_feature)
        vocab_dist = torch.softmax(output,dim = -1)

        if self.config.pointer_gen:
            vocab_dist_p = vocab_dist * genera_p
            context_dist_p = attention_score * (1 - genera_p)
            if oovs_zero is not None:
                vocab_dist_p = torch.cat([vocab_dist_p,oovs_zero],dim = -1)
            final_dist = vocab_dist_p.scatter_add(dim = -1,index=encoder_with_oov,src = context_dist_p)
        else:
            final_dist = vocab_dist


        return final_dist,next_decoder_status,current_context_vec,attention_score,genera_p,next_coverage




class PointerGeneratorNetworks(nn.Module):
    def __init__(self,config):

        # vob_size = 50000, embed_dim = 128, hidden_dim = 256, pad_idx = 0, dropout = 0.5, pointer_gen = True,
        # use_coverage = False, eps = 1e-12, coverage_loss_weight = 1.0, unk_token_idx = 1, start_token_idx = 2,
        # stop_token_idx = 3, max_decoder_len = 100, min_decoder_len = 35


        super(PointerGeneratorNetworks,self).__init__()
        self.all_mode = ["train","eval","decode"]

        self.config = config

        self.encoder = Encoder(config)

        self.w_h = nn.Linear(config.hidden_dim*2,config.hidden_dim*2,bias=False)     #

        self.reduce = Reduce(config)

        self.decoder_embed = BertEmbeddings(config)

        self.decoder = Decoder(config)

    # encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage, beam_size


    def forward(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,decoder_input = None,
                decoder_mask = None,decoder_target = None,mode = "train",start_tensor = None,beam_size = 4):

        assert mode in self.all_mode
        if mode in ["train","eval"]:
            return self._forward(encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,
                                 decoder_input,decoder_mask,decoder_target)
        elif mode in ["decode"]:
            return self._decoder(encoder_input = encoder_input,encoder_mask = encoder_mask,
                                 encoder_with_oov = encoder_with_oov,oovs_zero = oovs_zero,context_vec = context_vec,
                                 coverage = coverage,beam_size = beam_size)



    def _forward(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,
                 decoder_input,decoder_mask,decoder_target):



        encoder_outputs, encoder_hidden = self.encoder(encoder_input,encoder_mask)

        decoder_status = self.reduce(*encoder_hidden)

        encoder_features = self.w_h(encoder_outputs)


        batch_max_decoder_len = decoder_mask.size(1)
        assert batch_max_decoder_len <= self.config.max_title_len



        decoder_embed = self.decoder_embed(decoder_input)



        all_step_loss = []
        for step in range(1,batch_max_decoder_len+1):
            decoder_embed_one_step = decoder_embed[:,:step,:]
            decoder_mask_one_step = decoder_mask[:,:step]

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_embed_one_step=decoder_embed_one_step,
                    decoder_mask_one_step = decoder_mask_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs,
                    encoder_features = encoder_features,
                    encoder_mask=encoder_mask,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero,
                    encoder_with_oov=encoder_with_oov,
                    coverage=coverage,
                    step=step)

            target = decoder_target[:,step-1].unsqueeze(1)
            probs = torch.gather(final_dist,dim=-1,index=target).squeeze()
            step_loss = -torch.log(probs + self.config.eps)

            if self.config.use_coverage:
                coverage_loss = self.config.coverage_loss_weight * torch.sum(torch.min(attention_score, coverage), dim=-1)
                step_loss += coverage_loss
                coverage = next_coverage


            all_step_loss.append(step_loss)


        token_loss = torch.stack(all_step_loss, dim=1)

        decoder_mask_cut = decoder_mask[:,:batch_max_decoder_len].float()
        assert decoder_mask_cut.shape == token_loss.shape

        decoder_lens = decoder_mask.sum(dim=-1)

        token_loss_with_mask = token_loss * decoder_mask_cut
        batch_loss_sum_token = token_loss_with_mask.sum(dim=-1)
        batch_loss_mean_token = batch_loss_sum_token / decoder_lens.float()
        result_loss = batch_loss_mean_token.mean()

        return result_loss



    def _decoder(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,beam_size = 4):




        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)
        decoder_status = self.reduce(*encoder_hidden)

        encoder_features = self.w_h(encoder_outputs)


        beams = [Beam(tokens = [self.config.start_idx],log_probs = [1.0],status=decoder_status,context_vec = context_vec,
                      coverage = coverage)]

        step = 0
        result = []
        last_beam_size = 0
        current_beam_size = 1


        while step < self.config.max_title_len and len(result) < 4:

            # 当模型效果不太好时，可能会出现这种情况。有两个原因，
            # 1、模型生成的都是<unk>和<pad>人工过滤掉了
            # 2、标题最小长度小于规定长度。同样过滤掉了
            assert len(beams) != 0


            current_tokens_idx = [[token if token < self.config.vocab_size else self.config.unk_idx for token in b.tokens]
                                  for b in beams]
            decoder_input_one_step = torch.tensor(current_tokens_idx,dtype=torch.long,device=encoder_outputs.device)

            decoder_embed_one_step = self.decoder_embed(decoder_input_one_step)

            status_h_list = [b.status[0] for b in beams]
            status_c_list = [b.status[1] for b in beams]

            decoder_h = torch.cat(status_h_list,dim=1)   # status_h  (num_layers * num_directions, batch, hidden_size)
            decoder_c = torch.cat(status_c_list,dim=1)   # status_c  (num_layers * num_directions, batch, hidden_size)
            decoder_status = (decoder_h,decoder_c)

            context_vec_list = [b.context_vec for b in beams]
            context_vec = torch.cat(context_vec_list,dim=0)     # context_vec (batch,hidden*2)

            if self.config.use_coverage:
                coverage_list = [b.coverage for b in beams]
                coverage = torch.cat(coverage_list,dim=0)                 # coverage (batch,seq_len)
            else:
                coverage = None

            # 通常来说， 在step >= 1的条件下，这里的len(beams) == beam_size 的，
            # 但是也不排除模型效果不好的情况len(beams) < beam_size
            current_beam_size = len(beams)
            if current_beam_size != last_beam_size:
                last_beam_size = current_beam_size

                encoder_outputs_expand = encoder_outputs.expand(current_beam_size,encoder_outputs.size(1),
                                                                encoder_outputs.size(2))
                encoder_mask_expand = encoder_mask.expand(current_beam_size,encoder_mask.shape[1])
                encoder_features_expand = encoder_features.expand(current_beam_size,encoder_features.size(1),
                                                                  encoder_features.size(2))
                if oovs_zero is not None:
                    oovs_zero_expand = oovs_zero.expand(current_beam_size, oovs_zero.shape[1])
                else:
                    oovs_zero_expand = None
                encoder_with_oov_expand = encoder_with_oov.expand(current_beam_size, encoder_with_oov.shape[1])


            decoder_mask_one_step = torch.ones_like(decoder_input_one_step,device=decoder_input_one_step.device)

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_embed_one_step=decoder_embed_one_step,
                    decoder_mask_one_step = decoder_mask_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs_expand,
                    encoder_features=encoder_features_expand,
                    encoder_mask=encoder_mask_expand,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero_expand,
                    encoder_with_oov=encoder_with_oov_expand,
                    coverage=coverage,
                    step=step)

            # (batch_size,vob_size)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size * 2,dim=-1)

            all_beams = []
            for i in range(len(beams)):
                beam = beams[i]
                h_i = decoder_status[0][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                c_i = decoder_status[1][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                status_i = (h_i,c_i)
                context_vec_i = context_vec[i,:].unsqueeze(0)   # keep dim (batch,hidden*2)
                if self.config.use_coverage:
                    coverage_i = next_coverage[i,:].unsqueeze(0)  # keep dim (batch,seq_len)
                else:
                    coverage_i = None

                for j in range(beam_size*2):
                    if topk_ids[i,j] in [self.config.pad_idx,self.config.unk_idx]:
                        continue
                    new_beam = beam.update(token=topk_ids[i,j].item(),
                                           log_prob = topk_log_probs[i,j].item(),
                                           status=status_i,
                                           context_vec = context_vec_i,
                                           coverage = coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for beam in sort_beams(all_beams):
                if beam.tokens[-1] == self.config.stop_idx:
                    if len(beam.tokens) > self.config.min_title_len:
                        result.append(beam)
                    else:             # 如果以stop_token_idx 结尾，并且不够长，就丢弃掉，假如全部被丢掉了,0 == len(beams)
                        pass          # 把beam_size适当调大，min_decoder_len适当调小，如果还不行，模型估计废了。。。。。
                else:
                    beams.append(beam)
                if beam_size == len(beams) or len(result) == beam_size:
                    break
            step += 1

        if 0 == len(result):
            result = beams

        sorted_result = sort_beams(result)

        return sorted_result[0]





































































