#!/usr/bin/env python
# @Project ：NeuralCodeSum 
# @File    ：transformer2.py
# @Author  ：
# @Date    ：2022/9/10 14:49 
# 
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable
from c2nl.modules.char_embedding import CharEmbedding
from c2nl.modules.embeddings import Embeddings
from c2nl.modules.highway import Highway
from c2nl.encoders.transformer import TransformerEncoder
from c2nl.decoders.transformer import TransformerDecoder
from c2nl.inputters import constants
from c2nl.modules.global_attention import GlobalAttention
from c2nl.modules.copy_generator import CopyGenerator, CopyGeneratorCriterion
from c2nl.utils.misc import sequence_mask


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()

        self.enc_input_size = 0
        self.enc_input_size2 = 0
        self.dec_input_size = 0

        # at least one of word or char embedding options should be True
        assert args.use_src_word or args.use_src_char
        assert args.use_cfg_word or args.use_cfg_char
        assert args.use_tgt_word or args.use_tgt_char

        self.use_src_word = args.use_src_word
        self.use_cfg_word = args.use_cfg_word
        self.use_tgt_word = args.use_tgt_word
        if self.use_src_word:
            self.src_word_embeddings = Embeddings(args.emsize,
                                                  args.src_vocab_size,
                                                  constants.PAD)
            self.enc_input_size += args.emsize

        if self.use_cfg_word:
            self.cfg_word_embeddings = Embeddings(args.emsize,
                                                  args.cfg_vocab_size,
                                                  constants.PAD)
            self.enc_input_size2 += args.emsize

        if self.use_tgt_word:
            self.tgt_word_embeddings = Embeddings(args.emsize,
                                                  args.tgt_vocab_size,
                                                  constants.PAD)
            self.dec_input_size += args.emsize

        self.use_src_char = args.use_src_char
        self.use_cfg_char = args.use_cfg_char
        self.use_tgt_char = args.use_tgt_char
        if self.use_src_char:
            assert len(args.filter_size) == len(args.nfilters)
            self.src_char_embeddings = CharEmbedding(args.n_characters,
                                                     args.char_emsize,
                                                     args.filter_size,
                                                     args.nfilters)
            self.enc_input_size += sum(list(map(int, args.nfilters)))
            self.src_highway_net = Highway(self.enc_input_size, num_layers=2)
        if self.use_cfg_char:
            assert len(args.filter_size) == len(args.nfilters)
            self.cfg_char_embeddings = CharEmbedding(args.n_characters,
                                                     args.char_emsize,
                                                     args.filter_size,
                                                     args.nfilters)
            self.enc_input_size2 += sum(list(map(int, args.nfilters)))
            self.cfg_highway_net = Highway(self.enc_input_size2, num_layers=2)
        if self.use_tgt_char:
            assert len(args.filter_size) == len(args.nfilters)
            self.tgt_char_embeddings = CharEmbedding(args.n_characters,
                                                     args.char_emsize,
                                                     args.filter_size,
                                                     args.nfilters)
            self.dec_input_size += sum(list(map(int, args.nfilters)))
            self.tgt_highway_net = Highway(self.dec_input_size, num_layers=2)

        self.use_type = args.use_code_type
        if self.use_type:
            self.type_embeddings = nn.Embedding(len(constants.TOKEN_TYPE_MAP),
                                                self.enc_input_size)
        self.use_type2 = args.use_cfg_type
        if self.use_type2:
            self.type_embeddings2 = nn.Embedding(len(constants.TOKEN_TYPE_MAP),
                                                self.enc_input_size2)

        self.src_pos_emb = args.src_pos_emb
        self.cfg_pos_emb = args.cfg_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        # 改这里
        self.no_relative_pos = all(v == 0 for v in args.max_relative_pos)
        # self.no_relative_pos = False

        if self.src_pos_emb and self.no_relative_pos:
            self.src_pos_embeddings = nn.Embedding(args.max_src_len,
                                                   self.enc_input_size)

        if self.cfg_pos_emb and self.no_relative_pos:
            self.cfg_pos_embeddings = nn.Embedding(args.max_cfg_len,
                                                   self.enc_input_size2)

        if self.tgt_pos_emb:
            self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
                                                   self.dec_input_size)

        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self,
                sequence,
                sequence_char,
                sequence_type=None,
                mode='encoder',
                step=None):

        if mode == 'encoder1':
            word_rep = None
            if self.use_src_word:
                word_rep = self.src_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.use_src_char:
                char_rep = self.src_char_embeddings(sequence_char)  # B x P x f
                if word_rep is None:
                    word_rep = char_rep
                else:
                    word_rep = torch.cat((word_rep, char_rep), 2)  # B x P x d+f
                word_rep = self.src_highway_net(word_rep)  # B x P x d+f

            if self.use_type:
                type_rep = self.type_embeddings(sequence_type)
                word_rep = word_rep + type_rep

            if self.src_pos_emb and self.no_relative_pos:
                pos_enc = torch.arange(start=0,
                                       end=word_rep.size(1)).type(torch.LongTensor)
                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.src_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        elif mode == 'encoder2':
            word_rep = None
            if self.use_cfg_word:
                word_rep = self.cfg_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.use_cfg_char:
                char_rep = self.cfg_char_embeddings(sequence_char)  # B x P x f
                if word_rep is None:
                    word_rep = char_rep
                else:
                    word_rep = torch.cat((word_rep, char_rep), 2)  # B x P x d+f
                word_rep = self.cfg_highway_net(word_rep)  # B x P x d+f

            if self.use_type2:
                type_rep = self.type_embeddings(sequence_type)
                word_rep = word_rep + type_rep

            if self.cfg_pos_emb and self.no_relative_pos:
                pos_enc = torch.arange(start=0,
                                       end=word_rep.size(1)).type(torch.LongTensor)
                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.cfg_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        elif mode == 'decoder':
            word_rep = None
            if self.use_tgt_word:
                word_rep = self.tgt_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.use_tgt_char:
                char_rep = self.tgt_char_embeddings(sequence_char)  # B x P x f
                if word_rep is None:
                    word_rep = char_rep
                else:
                    word_rep = torch.cat((word_rep, char_rep), 2)  # B x P x d+f
                word_rep = self.tgt_highway_net(word_rep)  # B x P x d+f
            if self.tgt_pos_emb:
                if step is None:
                    pos_enc = torch.arange(start=0,
                                           end=word_rep.size(1)).type(torch.LongTensor)
                else:
                    pos_enc = torch.LongTensor([step])  # used in inference time

                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.tgt_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        else:
            raise ValueError('Unknown embedder mode!')

        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    def __init__(self,
                 args,
                 input_size):
        super(Encoder, self).__init__()

        self.transformer = TransformerEncoder(num_layers=args.nlayers,
                                              d_model=input_size,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                input,
                input_len):
        layer_outputs, _ = self.transformer(input, input_len)  # B x seq_len x h
        # 用所有层平均的权重还是只用最后一层输出的权重
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3), layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Decoder(nn.Module):
    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size

        self.split_decoder = args.split_decoder and args.copy_attn
        if self.split_decoder:
            # Following (https://arxiv.org/pdf/1808.07913.pdf), we split decoder
            self.transformer_c = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )
            self.transformer_d = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                dropout=args.trans_drop
            )

            # To accomplish eq. 19 - 21 from `https://arxiv.org/pdf/1808.07913.pdf`
            self.fusion_sigmoid = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.Sigmoid()
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.ReLU()
            )
        else:
            self.transformer = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )

        if args.reload_decoder_state:
            state_dict = torch.load(
                args.reload_decoder_state, map_location=lambda storage, loc: storage
            )
            self.decoder.load_state_dict(state_dict)

    def count_parameters(self):
        if self.split_decoder:
            return self.transformer_c.count_parameters() + self.transformer_d.count_parameters()
        else:
            return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len):

        if self.split_decoder:
            state_c = self.transformer_c.init_state(src_lens, max_src_len)
            state_d = self.transformer_d.init_state(src_lens, max_src_len)
            return state_c, state_d
        else:
            return self.transformer.init_state(src_lens, max_src_len)

    def decode(self,
               tgt_words,
               tgt_emb,
               memory_bank,
               state,
               step=None,
               layer_wise_coverage=None):

        if self.split_decoder:
            copier_out, attns = self.transformer_c(tgt_words,
                                                   tgt_emb,
                                                   memory_bank,
                                                   state[0],
                                                   step=step,
                                                   layer_wise_coverage=layer_wise_coverage)
            dec_out, _ = self.transformer_d(tgt_words,
                                            tgt_emb,
                                            memory_bank,
                                            state[1],
                                            step=step)
            f_t = self.fusion_sigmoid(torch.cat([copier_out[-1], dec_out[-1]], dim=-1))
            gate_input = torch.cat([copier_out[-1], torch.mul(f_t, dec_out[-1])], dim=-1)
            decoder_outputs = self.fusion_gate(gate_input)
            decoder_outputs = [decoder_outputs]
        else:
            decoder_outputs, attns = self.transformer(tgt_words,
                                                      tgt_emb,
                                                      memory_bank,
                                                      state,
                                                      step=step,
                                                      layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,
                memory_len,
                tgt_pad_mask,
                tgt_emb):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state)


class Transformer(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, args, tgt_dict):
        """"Constructor of the class."""
        super(Transformer, self).__init__()

        self.name = 'Transformer'
        self.use_two_encoder = True
        self.fuse_emb = False
        # 改这里
        # if len(args.max_relative_pos) != args.nlayers:
        #     assert len(args.max_relative_pos) == 1
        #     args.max_relative_pos = args.max_relative_pos * args.nlayers


        if type(args.max_relative_pos) != type(list()):
            args.max_relative_pos = [args.max_relative_pos] * args.nlayers
        else:
            if len(args.max_relative_pos) != args.nlayers:
                assert len(args.max_relative_pos) == 1
                args.max_relative_pos = args.max_relative_pos * args.nlayers
        print('args.max_relative_pos', args.max_relative_pos)

        self.embedder = Embedder(args)
        # if self.fuse_emb:
        #     self.embedder2 = Embedder(args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)

        self.encoder2 = Encoder(args, self.embedder.enc_input_size2)
        self.decoder = Decoder(args, self.embedder.dec_input_size)
        self.layer_wise_attn = args.layer_wise_attn

        self.generator = nn.Linear(self.decoder.input_size, args.tgt_vocab_size)
        if args.share_decoder_embeddings:
            if self.embedder.use_tgt_word:
                assert args.emsize == self.decoder.input_size
                self.generator.weight = self.embedder.tgt_word_embeddings.word_lut.weight

        self._copy = args.copy_attn
        if self._copy:
            self.copy_attn = GlobalAttention(dim=self.decoder.input_size,
                                             attn_type=args.attn_type)
            self.copy_generator = CopyGenerator(self.decoder.input_size,
                                                tgt_dict,
                                                self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=len(tgt_dict),
                                                    force_copy=args.force_copy)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _run_forward_ml(self,
                        code_word_rep,
                        code_char_rep,
                        code_type_rep,
                        code_len,
                        cfg_word_rep,
                        cfg_char_rep,
                        cfg_type_rep,
                        cfg_len,
                        summ_word_rep,
                        summ_char_rep,
                        summ_len,
                        tgt_seq,
                        src_map,
                        cfg_map,
                        alignment,
                        **kwargs):

        batch_size = code_len.size(0)
        # embed and encode the source sequence
        code_rep = self.embedder(code_word_rep,
                                 code_char_rep,
                                 code_type_rep,
                                 mode='encoder1')

        cfg_rep = self.embedder(cfg_word_rep,
                                cfg_char_rep,
                                cfg_type_rep,
                                mode='encoder2')

        if self.fuse_emb:
            # print(code_rep.size())
            fuse = torch.cat([code_rep, cfg_rep], dim=1)
            memory_bank, layer_wise_outputs = self.encoder(fuse, code_len)
            # memory_bank, layer_wise_outputs = self.encoder(fuse, code_len)
            enc_outputs = layer_wise_outputs if self.layer_wise_attn else memory_bank
        else:
            # 改
            memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # B x seq_len x h
            enc_outputs = layer_wise_outputs if self.layer_wise_attn else memory_bank
            # memory_bank, layer_wise_outputs = self.encoder2(cfg_rep, cfg_len)  # B x seq_len x h
            # enc_outputs = layer_wise_outputs if self.layer_wise_attn else memory_bank
        # print(memory_bank)

        # print(code_rep.size())
        # print(cfg_rep.size())
        # print(code_len)
        # print(cfg_len)

        if self.use_two_encoder:
            memory_bank2, layer_wise_outputs2 = self.encoder2(cfg_rep, cfg_len)
            enc_outputs2 = layer_wise_outputs2 if self.layer_wise_attn else memory_bank2
            # fuse = torch.cat((enc_outputs, enc_outputs2), dim=1).transpose(1,2)
            # linear = nn.Linear(fuse.size()[2], enc_outputs2.size()[1]).to(fuse.device)
            # enc_outputs = f.relu(linear(fuse)).transpose(1,2)
            # code_len = cfg_len

            # print(code_len, cfg_len)
            # 直接将两个编码器的encoder输出相加
            enc_outputs = torch.cat((enc_outputs, enc_outputs2), dim=1)
            # code_len += cfg_len

        # embed and encode the target sequence
        summ_emb = self.embedder(summ_word_rep,
                                 summ_char_rep,
                                 mode='decoder')
        summ_pad_mask = ~sequence_mask(summ_len, max_len=summ_emb.size(1))



        # 改了codelen
        if self.use_two_encoder:
            # print(code_len)
            # print(cfg_len)
            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                     cfg_len+code_len,
                                                     summ_pad_mask,
                                                     summ_emb)
        else:
            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                     code_len,
                                                     summ_pad_mask,
                                                     summ_emb)
        decoder_outputs = layer_wise_dec_out[-1]
        loss = dict()
        target = tgt_seq[:, 1:].contiguous()

        if self._copy:
            # copy_score: batch_size, tgt_len, src_len
            _, copy_score, _ = self.copy_attn(decoder_outputs,
                                              memory_bank,
                                              memory_lengths=code_len,
                                              softmax_weights=False)

            _, copy_score2, _ = self.copy_attn(decoder_outputs,
                                              memory_bank2,
                                              memory_lengths=cfg_len,
                                              softmax_weights=False)
            # print(decoder_outputs.size())
            # print(memory_bank.size(), memory_bank2.size())
            # mask copy_attn weights here if needed
            if kwargs['code_mask_rep'] is not None:
                mask = kwargs['code_mask_rep'].byte().unsqueeze(1)  # Make it broadcastable.
                copy_score.data.masked_fill_(mask, -float('inf'))
            if kwargs['cfg_mask_rep'] is not None:
                mask2 = kwargs['cfg_mask_rep'].byte().unsqueeze(1)  # Make it broadcastable.
                copy_score2.data.masked_fill_(mask2, -float('inf'))

            attn_copy = f.softmax(copy_score, dim=-1)
            scores = self.copy_generator(decoder_outputs, attn_copy, src_map)
            scores = scores[:, :-1, :].contiguous()
            attn_copy2 = f.softmax(copy_score2, dim=-1)
            scores2 = self.copy_generator(decoder_outputs, attn_copy2, cfg_map)
            scores2 = scores2[:, :-1, :].contiguous()
            # print(cfg_map, src_map)
            # print(cfg_map.size(), src_map.size())
            # print(scores.size(), scores2.size())
            ml_loss = self.criterion(scores,
                                     alignment[:, 1:].contiguous(),
                                     target)
            ml_loss2 = self.criterion(scores2,
                                     alignment[:, 1:].contiguous(),
                                     target)
            ml_loss = (ml_loss+ml_loss2)/2
            # print(ml_loss)
        else:
            scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
            ml_loss = self.criterion(scores.view(-1, scores.size(2)), target.view(-1))

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(constants.PAD).float())
        ml_loss = ml_loss.sum(1) * kwargs['example_weights']
        loss['ml_loss'] = ml_loss.mean()
        loss['loss_per_token'] = ml_loss.div((summ_len - 1).float()).mean()
        # print(loss)
        return loss

    def forward(self,
                code_word_rep,
                code_char_rep,
                code_type_rep,
                code_len,
                cfg_word_rep,
                cfg_char_rep,
                cfg_type_rep,
                cfg_len,
                summ_word_rep,
                summ_char_rep,
                summ_len,
                tgt_seq,
                src_map,
                cfg_map,
                alignment,
                **kwargs):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_char_rep: ``(batch_size, max_doc_len, max_word_len)``
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_char_rep: ``(batch_size, max_que_len, max_word_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(code_word_rep,
                                        code_char_rep,
                                        code_type_rep,
                                        code_len,
                                        cfg_word_rep,
                                        cfg_char_rep,
                                        cfg_type_rep,
                                        cfg_len,
                                        summ_word_rep,
                                        summ_char_rep,
                                        summ_len,
                                        tgt_seq,
                                        src_map,
                                        cfg_map,
                                        alignment,
                                        **kwargs)

        else:
            return self.decode(code_word_rep,
                               code_char_rep,
                               code_type_rep,
                               code_len,
                               cfg_word_rep,
                               cfg_char_rep,
                               cfg_type_rep,
                               cfg_len,
                               src_map,
                               cfg_map,
                               alignment,
                               **kwargs)

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):

        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([constants.BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        tgt_chars = None
        if self.embedder.use_tgt_char:
            tgt_chars = params['tgt_dict'].word_to_char_ids(constants.BOS_WORD)
            tgt_chars = torch.Tensor(tgt_chars.tolist()).unsqueeze(0)
            tgt_chars = tgt_chars.repeat(batch_size, 1)
            tgt_chars = tgt_chars.to(tgt_words).unsqueeze(1)

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []
        # 改
        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]

        if self.fuse_emb:
            dec_states = self.decoder.init_decoder(params['src_len']+params['cfg_len'], max_mem_len)
        else:
            dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len)


        attns = {"coverage": None}
        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn else params['memory_bank']


        if self.use_two_encoder:
            max_mem_len2 = params['memory_bank2'][0].shape[1] \
                if isinstance(params['memory_bank2'], list) else params['memory_bank2'].shape[1]
            dec_states2 = self.decoder.init_decoder(params['cfg_len'], max_mem_len2)
            enc_outputs2 = params['layer_wise_outputs2'] if self.layer_wise_attn else params['memory_bank2']
            # fuse = torch.cat([enc_outputs, enc_outputs2], dim=1).transpose(1,2)
            # linear = nn.Linear(fuse.size()[2], enc_outputs2.size()[1]).to(fuse.device)
            # enc_outputs = f.relu(linear(fuse)).transpose(1,2)
            # dec_states['src_max_len'] = dec_states2['src_max_len']
            # dec_states['src_len'] = dec_states2['src_len']
            # 这里可能有问题
            enc_outputs = torch.cat([enc_outputs, enc_outputs2], dim=1)
            dec_states['src_max_len'] = dec_states['src_max_len']+dec_states2['src_max_len']
            dec_states['src_len'] = dec_states['src_len']+dec_states2['src_len']

        # print(dec_states['src_max_len'])

        # +1 for <EOS> token
        for idx in range(params['max_len'] + 1):
            tgt = self.embedder(tgt_words,
                                tgt_chars,
                                mode='decoder',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            # print(decoder_outputs.size())
            if self._copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  params['memory_bank'],
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)
                _, copy_score2, _ = self.copy_attn(decoder_outputs,
                                                  params['memory_bank2'],
                                                  memory_lengths=params['cfg_len'],
                                                  softmax_weights=False)

                # mask copy_attn weights here if needed
                if params['src_mask'] is not None:
                    mask = params['src_mask'].byte().unsqueeze(1)  # Make it broadcastable.
                    copy_score.data.masked_fill_(mask, -float('inf'))
                if params['cfg_mask'] is not None:
                    mask2 = params['cfg_mask'].byte().unsqueeze(1)  # Make it broadcastable.
                    copy_score2.data.masked_fill_(mask2, -float('inf'))

                attn_copy = f.softmax(copy_score, dim=-1)
                attn_copy2 = f.softmax(copy_score2, dim=-1)

                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction2 = self.copy_generator(decoder_outputs,
                                                 attn_copy2,
                                                 params['cfg_map'])
                prediction = (prediction+prediction2)/2

                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = torch.LongTensor(params['blank'][b])
                        fill_b = torch.LongTensor(params['fill'][b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b, prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = f.softmax(prediction, dim=1)

                # print(prediction)
                # print()

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)

            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))
            if self._copy:
                mask = tgt.gt(len(params['tgt_dict']) - 1)
                copy_info.append(mask.float().squeeze(1))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            tgt_chars = None
            if self.embedder.use_tgt_char:
                tgt_chars = [params['tgt_dict'].word_to_char_ids(w).tolist() for w in words]
                tgt_chars = torch.Tensor(tgt_chars).to(tgt).unsqueeze(1)

            words = [params['tgt_dict'][w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, copy_info, dec_log_probs

    def decode(self,
               code_word_rep,
               code_char_rep,
               code_type_rep,
               code_len,
               cfg_word_rep,
               cfg_char_rep,
               cfg_type_rep,
               cfg_len,
               src_map,
               cfg_map,
               alignment,
               **kwargs):

        word_rep = self.embedder(code_word_rep,
                                 code_char_rep,
                                 code_type_rep,
                                 mode='encoder1')

        cfg_rep = self.embedder(cfg_word_rep,
                                cfg_char_rep,
                                cfg_type_rep,
                                mode='encoder2')
        if self.fuse_emb:
            fuse = torch.cat([word_rep, cfg_rep], dim=1)
            memory_bank, layer_wise_outputs = self.encoder(fuse, code_len)
        else:
            memory_bank, layer_wise_outputs = self.encoder(word_rep, code_len)  # B x seq_len x h
        memory_bank2, layer_wise_outputs2 = self.encoder2(cfg_rep, cfg_len)  # B x seq_len x h

        params = dict()
        params['memory_bank'] = memory_bank
        params['memory_bank2'] = memory_bank2
        params['layer_wise_outputs'] = layer_wise_outputs
        params['layer_wise_outputs2'] = layer_wise_outputs2
        params['src_len'] = code_len
        params['cfg_len'] = cfg_len
        params['source_vocab'] = kwargs['source_vocab']
        params['cfg_vocab'] = kwargs['cfg_vocab']
        params['src_map'] = src_map
        params['cfg_map'] = cfg_map
        params['src_mask'] = kwargs['code_mask_rep']
        params['cfg_mask'] = kwargs['cfg_mask_rep']
        params['fill'] = kwargs['fill']
        params['blank'] = kwargs['blank']
        params['src_dict'] = kwargs['src_dict']
        params['cfg_dict'] = kwargs['cfg_dict']
        params['tgt_dict'] = kwargs['tgt_dict']
        params['max_len'] = kwargs['max_len']
        params['src_words'] = code_word_rep
        params['cfg_words'] = cfg_word_rep

        dec_preds, attentions, copy_info, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        copy_info = torch.stack(copy_info, dim=1) if copy_info else None
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'copy_info': copy_info,
            'memory_bank': memory_bank,
            'attentions': attentions
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
