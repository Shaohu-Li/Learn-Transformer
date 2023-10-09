import math
import torch
import torch.nn as nn
# torch 中变量封装函数
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt

from transformer import *

if __name__ == "__main__":
    # 步骤一
    # # num_embeddings 表示你想要将多少种类进行编码
    # emb = nn.Embedding(12, 3, padding_idx=0)
    # input = torch.LongTensor([[1, 2, 3, 4], [4, 13, 2, 1]])
    # print(input.size())
    # print(emb(input).size())
    
    # 步骤二：
    # d_model = 512
    # d_vocab = 1024
    # x = Variable(torch.LongTensor([[100, 2, 421, 50], [300, 60, 2, 19]]))
    # emb = Embedding(d_model, d_vocab)
    # embx = emb(x)
    # 原本是 onehot 的话， embx.size() == (2, 4, max(x))
    # 现在是 [2, 4, 512]， 所以 nn.Embedding == W == onehot(x) * [(max(x), d_model])
    # 将原本稀疏的变量变成稠密变量，和 word2vec是一个样子，将 num_embeddings 理解为onehot 维度，也就是
    # 词向量的个数
    # print("embr : ", embx.size())
    
    # 步骤三
    # pe = PositionEmbedding(512)
    # print(pe.state_dict()['pe'].size()) 
    # print(pe.state_dict()) 
    
    # print(embx[0][0])
    # peEmbx = pe(embx)
    # print(peEmbx[0][0])
    
    # 计算 dropout 的比例
    # mask = peEmbx != 0
    # nozeros = mask.sum()
    
    # print(nozeros / mask.numel())
    
    # # 绘制位置编码
    # pe = PositionEmbedding(d_model=20)
    # y = pe.squeeze(0).numpy()
    
    # plt.figure(figsize=(20, 10))
    # # 显示 d_model 的随着 d_model 的不同的直的变化范围
    # plt.plot(np.arange(20), y[0, :])
    # # 绘制相同位置的i在不同的单词位置的上的变化范围
    # plt.plot(np.arange(100), y[:100, 4:8])
    # plt.legend(["word site", *["dim %d" % p for p in [4, 5, 6, 7]]])
    # plt.savefig("./位置编码显示.png")
    # plt.show()
    
    # 比较有位置编码和没有位置编码的区别：
    # x = Variable(torch.LongTensor([[100, 2, 421, 50] * 100]))
    # emb = Embedding(d_model=512, d_vocab=1024)
    # pe = PositionEmbedding(d_model=512)
    # embx = emb(x)
    # np_embx = embx[0][0].detach().numpy()
    # pe_embx = pe(embx)
    
    # np_pe_embx0 = pe_embx[0][0].detach().numpy()
    # np_pe_embx100 = pe_embx[0][99].detach().numpy()
    
    # plt.figure(figsize=(20, 10))
    # # 显示 d_model 的随着 d_model 的不同的直的变化范围
    # # plt.plot(np.arange(20), pe_embx[0, :])
    # # 绘制相同位置的i在不同的单词位置的上的变化范围
    # plt.plot(np.arange(512), np_embx[:])
    # plt.plot(np.arange(512), np_pe_embx0[:])
    # plt.plot(np.arange(512), np_pe_embx100[:])
    # plt.legend(["%s" % s for s in ["before", "after pos = 0", "after pos = 100"]])
    # plt.savefig("./位置编码显示.png")
    # plt.show()
    
    # x = subsequent_mask(size=20)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(x[0])
    # # print(x)
    # plt.show()
    
    # head = 8
    # # d_ff = 64
    # d_model = 512
    # p_dropout = 0.1
    # encoder_n = 8
    
    # # 测试注意力机制
    # x = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 13, 2, 1]]))
    # emb_x = Embedding(d_model, 1024)(x)
    # pe_emb_x = PositionEmbedding(d_model)(emb_x)
    
    # query = key = value = pe_emb_x
    # socres, attn = attention(query, key, value)
    # print(socres.size(), attn.size())
    # # print(attn[0], socres[0])
    
    # mask = subsequent_mask(size=4)
    # # print(mask)
    # socres_mask, attn_mask = attention(query, key, value, mask)
    # print(socres_mask.size(), attn_mask.size())
    # # print(attn_mask[0], socres_mask[0])
    
    # 测试多头注意力机制
    # mha = MultiHeadedAttention(head, d_model, p_dropout)
    # mha_result = mha(query, key, value, mask)
    # print("mha_result :", mha_result.size())
    
    # # 测试前馈全连接层
    # ff = PositionwiseFeedForward(d_model, 64, p_dropout)
    # ff_result = ff(mha_result)
    # print("ff result size : ", ff_result.size())
    
    # # 测试规范化层
    # ln = LayerNorm(d_model)
    # ln_result = ln(ff_result)
    # print("ln result size : ", ln_result.size())
    
    # # 测试sublayer
    # mha_sublayer = lambda x : mha(x, x, x, mask)
    # sublayer_con = SublayerConnection(d_model)
    # mha_sublayer_result = sublayer_con(pe_emb_x, mha_sublayer)
    # print("mha_sublayer_result size :", mha_sublayer_result.size())
    
    # # 测试 encoder 层 模块
    # encoder_layer = EncoderLayer(d_model, mha, ff, p_dropout)
    # encoder_layer_result = encoder_layer(query, mask)
    # print("encoder_result size : ", encoder_layer_result.size())
    
    # mask = None 
    # muti_attn = MultiHeadedAttention(head, d_model, p_dropout)
    # ff = PositionwiseFeedForward(d_model, d_ff, p_dropout)
    # encoder_layer = EncoderLayer(d_model, muti_attn, ff, p_dropout)

    # encoder = Encoder(encoder_layer, encoder_n)
    # encoder_result = encoder_layer(query, mask)
    # print("encoder_result size : ", encoder_result.size())
    
    # decode_mask = subsequent_mask(4)
    # self_attn = src_attn = MultiHeadedAttention(head, d_model, p_dropout)
    # de_ff = PositionwiseFeedForward(d_model, d_ff, p_dropout)
    # decoder_layer = DecoderLayer(d_model, self_attn, src_attn, de_ff, p_dropout)
    # # decoder_layer_re = decoder_layer(query, query, decode_mask, mask)
    # # print("decoder_layer_re size : ", decoder_layer_re.size())
    # decoder = Decoder(decoder_layer, 8)
    # de_re = decoder(query, encoder_result, decode_mask, mask)
    # print("de_re size : ", de_re.size())
    
    # head = Generator(d_model, 1000)
    # head_re = head(de_re)
    # print("head re size : ", head_re.size())
    
    # src_vocab = 100
    # tgt_vocab = 100
    # N = 6
    
    # x = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 13, 2, 1]]))
    # decode_mask = subsequent_mask(4)
    # transformer = make_model(src_vocab, tgt_vocab, N)
    # print(transformer(x, x, decode_mask, decode_mask).size())
    # torch.onnx.export(transformer, (x, x, decode_mask, decode_mask), "temp.onnx")
    pass