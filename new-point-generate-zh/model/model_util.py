


class Beam(object):
    def __init__(self,tokens,log_probs,status,context_vec,coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.status = status
        self.context_vec = context_vec
        self.coverage = coverage

    def update(self,token,log_prob,status,context_vec,coverage):
        return Beam(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + [log_prob],
            status = status,
            context_vec = context_vec,
            coverage = coverage
        )


    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


def sort_beams(beams):
    return sorted(beams, key=lambda beam:beam.avg_log_prob, reverse=True)





class ModelConfig(object):
    def __init__(self,**kwargs):

        self.attention_probs_dropout_prob = kwargs["attention_probs_dropout_prob"]
        self.hidden_dropout_prob = kwargs["hidden_dropout_prob"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.initializer_range = kwargs["initializer_range"]
        self.intermediate_dim = kwargs["intermediate_dim"]
        self.max_content_len = kwargs["max_content_len"]
        self.max_title_len = kwargs["max_title_len"]
        self.min_title_len = kwargs["min_title_len"]
        self.num_attention_heads = kwargs["num_attention_heads"]
        self.vocab_size = kwargs["vocab_size"]
        self.eps = kwargs["eps"]
        self.use_coverage = kwargs["use_coverage"]
        self.pointer_gen = kwargs["pointer_gen"]
        self.coverage_loss_weight = kwargs["coverage_loss_weight"]
        self.encoder_lstm_num_layer = kwargs["encoder_lstm_num_layer"]
        self.decoder_lstm_num_layer = kwargs["decoder_lstm_num_layer"]




