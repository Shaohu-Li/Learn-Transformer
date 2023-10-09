import math
import torch
import torch.nn as nn
# torch 中变量封装函数
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt

from copy import deepcopy as c

class Embedding(nn.Module):
    def __init__(self, d_model, d_vocab):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(d_vocab, d_model)
    
    def forward(self, x):
        # 其中 math.sqrt(self.d_model) 为缩放因子 
        return self.emb(x) * math.sqrt(self.d_model)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model=512, p_dropout=0.1, max_len=5000):
        super().__init__()
        # 对输入进行随即丢弃
        self.dropout = nn.Dropout(p=p_dropout)
        
        # 先初始化一个全为 0 的位置编码
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len).unsqueeze(1) # => (max_len, 1)
        # 进行正弦和余弦的位置编码
        # pe(pos, 2i)     = sin(pos/10000^(2i/d_model)) = sin(pos * e^(2i * -log(10000.0)/ d_model ))
        # pe(pos, 2i + 1) = cos(pos/10000^(2i/d_model)) = cos(pos * e^(2i * -log(10000.0) / d_model ))
        # 怎么理解 pos 和 i， ==》 pos 代表第输入中单词在pos位置， 而 i 代表词向量中的位置
        # torch.arange(0, d_model, 2) 对应 2i
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 此时 pe = (max_len, d_model)，因为我们的输入为三维，带有 batch_size
        pe = pe.unsqueeze(0)
        
        # 最后将 pe 注册为模型的 buffer，什么是 buffer 呢？
        # 我们把它认为是对模型有帮助的,但是却不是模型中的超参数,不需要随着优化步骤进行更新增益对象
        # 注册之后,我们就可以在模型保存后重加载时和模型结构参数一同被记载,
        # 可以被认为是绑定到我们的模型的一些不优化的参数 -> state_dict() 中
        # 解释 : https://blog.csdn.net/weixin_46197934/article/details/119518497
        self.register_buffer('pe', pe)
    def forward(self, x):
        # 将当前输入的[batch, word_len, d_model] + pe
        # 为什么 :x.size(1),因为我们提前算好 max_len,方便一点
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
def subsequent_mask(size=512):
    """
    生成向后的掩盖的掩码张量， 参数 size 是掩码张量的最后两个维度的大小，
    它的最后两维形成一个方阵
    1 0 0 0
    1 1 0 0
    1 1 1 0
    1 1 1 1
    可以理解为前面的输入没必要和后面还没有输入的地方做注意力机制，
    """
    # 先形成全1矩阵，再变换为上三角矩阵，最后1减去就变成了下三角矩阵了
    subsequent_mask = np.triu(np.ones((1, size, size)), k = 1).astype(np.int8)
    return torch.from_numpy(1 - subsequent_mask)

def attention(query, key, value, mask=None, dropout=None):
    """
    计算公式为： ans = softmax(Q * K_T / sqrt(d_k)) * V
    """
    # 计算注意力机制的分数
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 是否存在掩码张量
    if mask is not None:
        # 将 mask 为 0 的地方设置为 -1e9
        scores = scores.masked_fill(mask == 0, -1e9)

    # 计算 softmax
    p_attn = F.softmax(scores, dim=-1)
    
    # 是否存在 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 返回注意力机制的值和注意力机制的分数, 并返回注意力机制的值
    return torch.matmul(p_attn, value), p_attn

def clone_modules(module, N):
    """
    克隆 N 个 module
    """
    return nn.ModuleList([c(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    """
    实现多头注意力机制
    """
    def __init__(self, head=8, embedding_dim=512, p_dropout=0.1):
        super().__init__()
        #  确定多头的head需要整除词嵌入的维度
        assert embedding_dim % head == 0
        
        # 确认每个头的词嵌入的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        
        # 获得线性层， 需要获得4个，分别是 Q、K、V 和最后的输出
        self.linears = clone_modules(nn.Linear(embedding_dim, embedding_dim), 4)
        
        # 初始化注意力张量
        self.attn = None
        
        # 初始化 dropout 对象
        self.dropout = nn.Dropout(p=p_dropout)
    
    def forward(self, query, key, value, mask=None):
        # 是否存在 mask 
        if mask is not None:
            # 因为是多头注意力机制，所以这里需要将 mask 扩展维度
            mask = mask.unsqueeze(0)
            
        # 接着，我们获得一个batch_size的变量， 代表有多少条样本
        batch_size = query.size(0)
        
        # view 是为了让 Q、K、V变成多头注意力的形式， 但是这样的形式，是没有办法输入到attention 中
        # 进行并行处理的， 如果把 head 和 词数量 的位置变化一下，就是每个头单独进行注意力计算
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]
        
        # 将多个头的输出送入到 attention 中一起并行计算注意力即可
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 得到多头注意力的结果之后，我们还需要转化一下维度，拼凑为原始的 d_model 的注意力机制
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        
        # 最后输出的时候，再经过一个线性变换层即可
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    """
    前馈全连接层
    return linear(dropout(relu(linear(x))))
    """
    def __init__(self, d_model, d_ff, p_dropout=0.1):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_ff)
        self.f2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=p_dropout)
        
    def forward(self, x):
        return self.f2(self.dropout(F.relu(self.f1(x))))
    
class LayerNorm(nn.Module):
    """
    对单个batch、单个样本中的最后特征维度进行归一化操作，
    解决NLP中 BN 中输入样本长度不一致的问题
    """
    def __init__(self, features_size, eps=1e-6):
        super().__init__()
        self.train_mean = nn.Parameter(torch.ones(features_size))
        self.train_std = nn.Parameter(torch.zeros(features_size))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.train_mean + self.train_std
    
class SublayerConnection(nn.Module):
    """
    对照着模型进行残差连接
    """
    def __init__(self, size, p_dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=p_dropout)
        
    def forward(self, x, sublayer):
        # 在论文复现的时候发现 self.norm 放在 里面 比放在 外面 好
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, attn_layer, feed_forward_layer, dropout):
        super().__init__()
        self.d_model = d_model
        self.attn = attn_layer
        self.feed_forward = feed_forward_layer
        
        # 还需要初始化两个残差连接
        self.subLayers = clone_modules(SublayerConnection(d_model, dropout), 2)
        
    def forward(self, x, mask):
        # 经过多头注意力层，然后残差，然后前馈层，然后再残差
        x = self.subLayers[0](x, lambda x : self.attn(x, x, x, mask))
        return self.subLayers[1](x, self.feed_forward)
    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.norm = LayerNorm(layer.d_model)
        self.layers = clone_modules(layer, N)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return  self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, p_dropout) -> None:
        super().__init__()
        # self_attn 为自注意力机制 src_attn 为原始的多头注意力机制
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff = feed_forward
        self.subLayers = clone_modules(SublayerConnection(d_model, p_dropout), 3)
        
    def forward(self, x, m, src_mask, tgt_mask):
        """
        x: 上一层的输入， m(memory): 来自编码器的输出， src_mask: 原数据掩码，tgt_mask: 目标数据掩码
        """
        # 第一步，是解码器的输入自己和自己作注意力机制，这个时候哦，我们不希望前面已经输出的和后面的词作注意力机制
        # 因为解码器端口的输入是我们一次给完整的，方便我们计算损失和并行化
        x = self.subLayers[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))
        # 第二步：是解码器输入得到的注意力结果之后，和我们编码器的最终的输出进行注意力的操作，
        # 这里的 src_mask 并不是因为抑制信息泄漏，而是屏蔽对结果没有意义的字符而产生的注意力的值，以此提升模型效果和训练速度（输入中无用的字符？）
        x = self.subLayers[1](x, lambda x : self.self_attn(x, m, m, src_mask))
        
        return self.subLayers[2](x, self.ff)

class Decoder(nn.Module):
    def  __init__(self, layer, N):
        super().__init__()
        self.norm = LayerNorm(layer.d_model)
        self.layers = clone_modules(layer, N)
        
    def forward(self, x, m, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, m, tgt_mask, src_mask)
            
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # log_softmax 和 softmax 对于最终的输出是没有影响的，
        # 但是可以解决 softmax 数值不稳定的现象
        return F.log_softmax(self.project(x), -1)
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.generator(
                self.decode(
                    self.encode(src, src_mask), src_mask, tgt, tgt_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, m, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), m, src_mask, tgt_mask)
    
# 构建用于 transformer 的模型
def make_model(src_vocab, tgt_vocab, N=6, 
                d_model=512, d_ff=2048, head=8, p_dropout=0.1):
    # 初始化一些需要公用的层,后面使用deepcopy,
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, p_dropout)
    pos_layer = PositionEmbedding(d_model, p_dropout)
    
    # 初始化我们的模型
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), p_dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), p_dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(pos_layer)),
        nn.Sequential(Embedding(d_model, src_vocab), c(pos_layer)),
        Generator(d_model, tgt_vocab)
    )
    
    # 初始化那些参数维度大于一的，将其初始化为服从均匀分布的矩阵。显示的设置模型参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model