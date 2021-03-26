import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from utils.fc import FCNet, get_norm, FCNet_sigmoid
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
import math
from allennlp.modules.matrix_attention import BilinearMatrixAttention

def pad_sequence(sequence, tags_attention):
    """
    :param sequence: [\sum b, .....] sequence
    :param tags_attention: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(tags_attention.shape[0], 4, tags_attention.shape[2], sequence.shape[-1])
    lengths = tags_attention.sum(-1).tolist()
    start = 0
    for i, lengths_1 in enumerate(lengths):
        for j, diff in enumerate(lengths_1):
            if diff > 0:
                output[i, j, :diff] = sequence[start:(start + diff)]
            else:
                raise ValueError("Oh no! tags_attention has zero tags!")
            start += diff
    return output

# Default concat, 1 layer, output layer
class Att_0_UDBU(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_0_UDBU, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, 1)
        w = masked_softmax(logits, box_mask[:, None], dim=-1)
        return w

    def logits(self, v, q):
        batch_v, num_obj, v_dim = v.size()
        q = q.repeat(1, 1, num_obj, 1)
        v = v.unsqueeze(1).repeat(1, 4, 1, 1)
        vq = torch.cat((v, q), -1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# Default concat, 1 layer, output layer
class Att_0_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_0_layer1, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, 1)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# Default concat, 1 layer, output layer
class Att_0_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_0_layer2, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        q_proj = q.repeat(1, 1, num_token, 1)
        vq = torch.cat((v, q_proj), -1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_0_layer2_keycat_textual_visual(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_0_layer2_keycat_textual_visual, self).__init__()
        # norm_layer = get_norm(norm)
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        q_proj = q.repeat(1, 1, num_token, 1)
        vq = torch.cat((v, q_proj), -1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# concat, 2 layer, output layer
class Att_1_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_1_layer1, self).__init__()
        self.nonlinear_1 = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_2 = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), -1)
        joint_repr_1 = self.nonlinear_1(vq)
        joint_repr_2 = self.nonlinear_2(joint_repr_1)
        logits = self.linear(joint_repr_2).squeeze(-1)
        return logits

# concat, 2 layer, output layer
class Att_1_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_1_layer2, self).__init__()
        self.nonlinear_1 = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_2 = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        q_proj = q.repeat(1, 1, num_token, 1)
        vq = torch.cat((v, q_proj), -1)
        joint_repr_1 = self.nonlinear_1(vq)
        joint_repr_2 = self.nonlinear_2(joint_repr_1)
        logits = self.linear(joint_repr_2).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_1_layer2_keycat_textual_visual(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_1_layer2_keycat_textual_visual, self).__init__()
        self.nonlinear_1 = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_2 = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        q_proj = q.repeat(1, 1, num_token, 1)
        vq = torch.cat((v, q_proj), -1)
        joint_repr_1 = self.nonlinear_1(vq)
        joint_repr_2 = self.nonlinear_2(joint_repr_1)
        logits = self.linear(joint_repr_2).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, output layer
class Att_2_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_2_layer1, self).__init__()
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        batch_v, num_obj, v_dim = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, num_obj, 1) # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, output layer
class Att_2_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Att_2_layer2, self).__init__()
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).repeat(1, 1, num_token, 1) # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_3_layer1, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, -2)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        batch_v, num_obj, v_dim = v.size()
        v_proj = self.v_proj(v)# [batch, num_obj, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, num_obj, 1) # [batch, num_obj, num_hid]
        # v_proj = self.v_proj(v).unsqueeze(1).unsqueeze(1).expand(-1, k, num_token, -1, -1)
        # q_proj = self.q_proj(q).unsqueeze(3).expand(-1, -1, -1, num_obj, -1)  # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_3_layer2, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q).repeat(1, 1, num_token, 1) # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3_layer2_keycat_textual_visual(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_3_layer2_keycat_textual_visual, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q).repeat(1, 1, num_token, 1)  # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer   原始att3
class Att_3_layer1_beifen(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_3_layer1_beifen, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)
        self.num_hid = num_hid

    def forward(self, v, q, box_mask):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, -2)
        w = masked_softmax(logits, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        batch_q, k, num_token, _ = q.size()
        batch_v, num_obj, v_dim = v.size()
        v_proj = self.v_proj(v).unsqueeze(1).unsqueeze(1).repeat(1, k, num_token, 1, 1) # [batch, k, num_token, num_obj, num_hid]
        q_proj = self.q_proj(q).unsqueeze(3).repeat(1, 1, 1, num_obj, 1) # [batch, k, num_token, num_obj, num_hid]
        # v_proj = self.v_proj(v).unsqueeze(1).unsqueeze(1).expand(-1, k, num_token, -1, -1)
        # q_proj = self.q_proj(q).unsqueeze(3).expand(-1, -1, -1, num_obj, -1)  # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits



# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3S_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_3S_layer1, self).__init__()
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        logits_batch = pad_sequence(logits, tags_attention)
        w_ = nn.functional.sigmoid(logits_batch)
        w = w_ * box_mask[:, None, None].float()
        return w

    def logits(self, v, q):
        batch_v, num_obj, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, num_obj, 1)  # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_4_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer1, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, -2)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        v_proj = self.v_proj(v)# [batch, num_obj, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1)
        # v_proj = self.v_proj(v).unsqueeze(1).unsqueeze(1).expand(-1, k, num_token, -1, -1)
        # q_proj = self.q_proj(q).unsqueeze(3).expand(-1, -1, -1, num_obj, -1)  # [batch, k, num_token, num_obj, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(1)
        return logits


# 1 layer seperate, element-wise *, 1 layer seperate, output layer
# 该attention处理细节还是回归到老的attention计算方式，为的是省显存，这里的batchsize就是传入模型的batchsize，不再是tag的数量
class Att_4_layer1_save_memory(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer1_save_memory, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        # self.v_linear = nn.Linear(num_hid, num_hid)
        # self.q_linear = nn.Linear(num_hid, num_hid)
        # self.v_proj = torch.nn.Sequential(
        #     nn.Linear(v_dim, num_hid),
        #     torch.nn.Dropout(0.3, inplace=False)
        # )
        # self.q_proj = torch.nn.Sequential(
        #     nn.Linear(q_dim, num_hid),
        #     torch.nn.Dropout(0.3, inplace=False)
        # )
        self.num_hid = num_hid

    def forward(self, v, q, box_mask):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, -2)
        # logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        v_proj = self.v_proj(v).unsqueeze(1)# [batch, 1, num_obj, num_hid]
        q_proj = self.q_proj(q) # [batch, 4, num_token, num_hid]
        # v_proj = self.v_linear(v_proj)
        # q_proj = self.q_linear(q_proj)
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        # logits = logits_.squeeze(1)
        logits = logits_
        return logits

# 该attention处理细节还是回归到老的attention计算方式，为的是省显存，这里的batchsize就是传入模型的batchsize，不再是tag的数量
class Att_4_layer1_proj_dropout_like_transflrmer(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer1_proj_dropout_like_transflrmer, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = nn.Linear(v_dim, num_hid)
        self.q_proj = nn.Linear(q_dim, num_hid)
        self.num_hid = num_hid
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w_ = masked_softmax(logits, box_mask[:, None, None], dim=-1)
        w = self.dropout(w_)
        return w

    def logits(self, v, q):
        v_proj = self.v_proj(v).unsqueeze(1)# [batch, 1, num_obj, num_hid]
        q_proj = self.q_proj(q) # [batch, 4, num_token, num_hid]
        logits = torch.matmul(q_proj, v_proj.transpose(-2, -1))/math.sqrt(self.num_hid)
        return logits

class Att_4_layer1_additive_save_memory(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer1_additive_save_memory, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = nn.Linear(v_dim, num_hid)
        self.q_proj = nn.Linear(q_dim, num_hid)
        self.linear = nn.Linear(num_hid, 1)
        self.num_hid = num_hid

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, -2)
        # logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        v_proj = self.v_proj(v).unsqueeze(1)# [batch, 1, num_obj, num_hid]
        q_proj = self.q_proj(q) # [batch, 4, num_token, num_hid]
        # v_proj = self.v_proj(v).unsqueeze(1).unsqueeze(1).expand(-1, k, num_token, -1, -1)
        # q_proj = self.q_proj(q).unsqueeze(3).expand(-1, -1, -1, num_obj, -1)  # [batch, k, num_token, num_obj, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(1) / math.sqrt(self.num_hid)
        joint_repr = v_proj + q_proj
        print(v_proj)
        print(q_proj)
        print(joint_repr)
        joint_repr = F.tanh(joint_repr)
        print(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_4_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer2, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(2)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
# 该attention处理细节还是回归到老的attention计算方式，为的是省显存，这里的batchsize就是传入模型的batchsize，不再是tag的数量
# 该处采用类似transformer的mask方式，也就是将应该mask掉的地方替换为一个很大的负数（-1e9），使得该处的softmax值为0
class Att_4_layer1_transformer_mask(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer1_transformer_mask, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        scores = logits.masked_fill(box_mask[:, None, None] == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return p_attn

    def logits(self, v, q):
        v_proj = self.v_proj(v).unsqueeze(1)# [batch, 1, num_obj, num_hid]
        q_proj = self.q_proj(q) # [batch, 4, num_token, num_hid]
        # v_proj = self.v_proj(v).unsqueeze(1).unsqueeze(1).expand(-1, k, num_token, -1, -1)
        # q_proj = self.q_proj(q).unsqueeze(3).expand(-1, -1, -1, num_obj, -1)  # [batch, k, num_token, num_obj, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
# 该attention处理细节还是回归到老的attention计算方式，为的是省显存，这里的batchsize就是传入模型的batchsize，不再是tag的数量
# 这里将需要mask的地方加上一个很大的负数（-1e9），使得该处softmax值为0
class Att_4_layer1_huge_negative_mask(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer1_huge_negative_mask, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        huge_negative = torch.zeros_like(logits).masked_fill(box_mask[:, None, None] == 0, -1e9)
        scores = logits + huge_negative
        p_attn = F.softmax(scores, dim=-1)
        return p_attn

    def logits(self, v, q):
        v_proj = self.v_proj(v).unsqueeze(1)# [batch, 1, num_obj, num_hid]
        q_proj = self.q_proj(q) # [batch, 4, num_token, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_4_layer2_transformer_mask(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer2_transformer_mask, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        scores = logits.masked_fill(tags_attention == 0, -1e9)
        att2 = F.softmax(scores, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(2)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_4_layer2_huge_negative_mask(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer2_huge_negative_mask, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        huge_negative = torch.zeros_like(logits).masked_fill(tags_attention == 0, -1e9)
        scores = logits + huge_negative
        att2 = F.softmax(scores, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(2)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer，对不同的第一层attention，使用不同的query值去计算其第二层attention的权重
class Att_4S_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4S_layer2, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer，对不同的第一层attention，使用不同的query值去计算其第二层attention的权重
class Att_40_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_40_layer2, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        vq = torch.cat((v_proj, q_proj), -1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_4_layer2_keycat_textual_visual(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer2_keycat_textual_visual, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        # self.v_linear = nn.Linear(num_hid, num_hid)
        # self.q_linear = nn.Linear(num_hid, num_hid)
        # self.v_proj = torch.nn.Sequential(
        #     nn.Linear(v_dim, num_hid),
        #     torch.nn.Dropout(0.3, inplace=False)
        # )
        # self.q_proj = torch.nn.Sequential(
        #     nn.Linear(q_dim, num_hid),
        #     torch.nn.Dropout(0.3, inplace=False)
        # )
        self.num_hid = num_hid

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        # v_proj = self.v_linear(v_proj)
        # q_proj = self.q_linear(q_proj)
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))
        logits = logits_.squeeze(2)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_4_layer2_keycat_textual_visual_proj_dropout_like_transflrmer(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer2_keycat_textual_visual_proj_dropout_like_transflrmer, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = nn.Linear(v_dim, num_hid)
        self.q_proj = nn.Linear(q_dim, num_hid)
        self.num_hid = num_hid
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2_ = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        att2 = self.dropout(att2_)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q)
        logits_ = torch.matmul(q_proj, v_proj.transpose(-2, -1))/math.sqrt(self.num_hid)
        logits = logits_.squeeze(2)
        return logits

class Att_4_layer2_additive_keycat_textual_visual(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_4_layer2_additive_keycat_textual_visual, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = nn.Linear(v_dim, num_hid)
        self.q_proj = nn.Linear(q_dim, num_hid)
        self.linear = nn.Linear(num_hid, 1)
        self.num_hid = num_hid

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q) # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj + q_proj
        joint_repr = F.tanh(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits

class Att_Bilinear_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_Bilinear_layer1, self).__init__()
        # norm_layer = get_norm(norm)
        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=q_dim,
            matrix_2_dim=v_dim,
            # matrix_2_dim=512
        )
        self.num_hid = num_hid

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, -2)
        # logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        logits = self.obj_attention(q.view(q.shape[0], q.shape[1]*q.shape[2], -1), v).view(q.shape[0], q.shape[1], q.shape[2], v.shape[1])
        return logits

class Att_Bilinear_layer2_keycat_textual_visual(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_Bilinear_layer2_keycat_textual_visual, self).__init__()
        # norm_layer = get_norm(norm)
        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=q_dim,
            matrix_2_dim=v_dim,
            # matrix_2_dim=512
        )
        self.num_hid = num_hid

    def forward(self, q, att1, obj_reps, tags_attention, t_rep, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_ = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        # v = torch.cat((v_, t_rep), dim=-1)
        v = torch.add(v_, 1, t_rep)
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        logits = self.obj_attention(q.view(q.shape[0]*q.shape[1], q.shape[2], -1), v.view(v.shape[0]*v.shape[1], v.shape[2], -1)).view(q.shape[0], q.shape[1], q.shape[2], v.shape[2]).squeeze(2)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer'
# 这里感觉3S未必适合于做第二层attention，因为这里会出现一个问题，就是就算句子中只有一个tag，那模型也会对这一个tag对应的attention进行sigmoid权值计算，计算出来的结果就不再是1了，这样有一定的不合理性，但结果不一定差
class Att_3S_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_3S_layer2, self).__init__()
        # norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], output_dropout=0.3)
        self.q_proj = FCNet([q_dim, num_hid], output_dropout=0.3)
        self.nonlinear = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2_ = nn.functional.sigmoid(logits)
        att2 = att2_ * tags_attention.float()
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_token, num_hid]
        q_proj = self.q_proj(q).repeat(1, 1, num_token, 1) # [batch, k, num_token, num_obj, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr).squeeze(-1)
        return logits


# concat w/ 2 layer seperate, element-wise *, output layer
class Att_PD_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_PD_layer1, self).__init__()
        self.nonlinear_1 = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_2 = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.nonlinear_gate_1 = FCNet_sigmoid([v_dim + q_dim, num_hid])
        self.nonlinear_gate_2 = FCNet_sigmoid([num_hid, num_hid])
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        batch_v, num_obj, v_dim = v.size()
        q = q.unsqueeze(1).repeat(1, num_obj, 1)
        vq = torch.cat((v, q), 2)
        joint_repr_1 = self.nonlinear_1(vq)
        joint_repr = self.nonlinear_2(joint_repr_1)
        gate_1 = self.nonlinear_gate_1(vq)
        gate = self.nonlinear_gate_2(gate_1)
        logits = joint_repr*gate
        logits = self.linear(logits).squeeze(-1)
        return logits

# concat w/ 2 layer seperate, element-wise *, output layer
class Att_PD_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_PD_layer2, self).__init__()
        self.nonlinear_1 = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_2 = FCNet([num_hid, num_hid], output_dropout=0.3)
        self.nonlinear_gate_1 = FCNet_sigmoid([v_dim + q_dim, num_hid])
        self.nonlinear_gate_2 = FCNet_sigmoid([num_hid, num_hid])
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        q = q.repeat(1, 1, num_token, 1)
        vq = torch.cat((v, q), -1)
        joint_repr_1 = self.nonlinear_1(vq)
        joint_repr = self.nonlinear_2(joint_repr_1)
        gate_1 = self.nonlinear_gate_1(vq)
        gate = self.nonlinear_gate_2(gate_1)
        logits = joint_repr*gate
        logits = self.linear(logits).squeeze(-1)
        return logits


# concat w/ 1 layer seperate, element-wise *, output layer
# concat w/ 2 layer seperate, element-wise *, output layer
class Att_P_layer1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_P_layer1, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_gate = FCNet_sigmoid([v_dim + q_dim, num_hid])
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q, box_mask, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        logits_batch = pad_sequence(logits, tags_attention)
        w = masked_softmax(logits_batch, box_mask[:, None, None], dim=-1)
        return w

    def logits(self, v, q):
        batch_v, num_obj, v_dim = v.size()
        q = q.unsqueeze(1).repeat(1, num_obj, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        gate = self.nonlinear_gate(vq)
        logits = joint_repr*gate
        logits = self.linear(logits).squeeze(-1)
        return logits

# concat w/ 2 layer seperate, element-wise *, output layer
class Att_P_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_P_layer2, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], output_dropout=0.3)
        self.nonlinear_gate = FCNet_sigmoid([v_dim + q_dim, num_hid])
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, q, att1, obj_reps, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v = torch.einsum('bnao,bod->bnad', [att1, obj_reps])
        logits = self.logits(v, q)
        att2 = masked_softmax(torch.div(logits, t), tags_attention, dim=-1)
        attention_last = torch.matmul(att2[:, :, None, :], att1).squeeze(2)
        return attention_last

    def logits(self, v, q):
        batch_v, k, num_token, v_dim = v.size()
        q = q.repeat(1, 1, num_token, 1)
        vq = torch.cat((v, q), -1)
        joint_repr = self.nonlinear(vq)
        gate = self.nonlinear_gate(vq)
        logits = joint_repr*gate
        logits = self.linear(logits).squeeze(-1)
        return logits

# 第二层attention采用平均操作，这里要注意的是避开把mask以外的全零attention也参与到mean计算中来
class Att_mean_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_mean_layer2, self).__init__()
        pass

    def forward(self, att1, tags_attention, t=1):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        att1_mask = att1 * tags_attention[:, :, :, None].float()
        att1_mask_sum = att1_mask.sum(2)
        tags_attention_sum = tags_attention.sum(-1).float()
        att1_mask_mean = torch.div(att1_mask_sum, tags_attention_sum.unsqueeze(-1))
        return att1_mask_mean


# 第二层attention采用平均操作，这里要注意的是避开把mask以外的全零attention也参与到mean计算中来
class Att_sum_layer2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm='weight', act='LeakyReLU', dropout=0.3):
        super(Att_sum_layer2, self).__init__()
        pass

    def forward(self, att1, tags_attention):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        att1_mask = att1 * tags_attention[:, :, :, None].float()
        att1_mask_sum = att1_mask.sum(2)
        return att1_mask_sum



class Att_DP(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.query_proj = FCNet([query_size, hidden_size], output_dropout=0.3)
        self.key_proj = FCNet([key_size, hidden_size], output_dropout=0.3)

    def forward(self, query, key, query_mask=None, key_mask=None):
        """shape of input and mask should be the same in first several dimensions"""
        logits = self.logits(query, key)
        if key_mask is not None:
            w = masked_softmax(logits, key_mask.unsqueeze(-2), dim=-1)
        else:
            w = torch.softmax(logits, dim=-1)
        if query_mask is not None:
            w = w * query_mask.unsqueeze(-1)  # mask padding in sentence
        return w

    # return attended key, attention weights and attention logits
    def forward_with_logits(self, query, key, query_mask=None, key_mask=None):
        logits = self.logits(query, key)
        if key_mask is not None:
            w = masked_softmax(logits, key_mask.unsqueeze(-2), dim=-1)
        else:
            w = torch.softmax(logits, dim=-1)
        if query_mask is not None:
            w = w * query_mask.unsqueeze(-1)  # mask padding in sentence
        value = w.matmul(key)
        return value, (w, logits)

    def logits(self, query, key):        # 柯洁说，再说吧！！！！！！！！！！！
        query = self.query_proj(query)
        key = self.key_proj(key)
        logits = query.matmul(key.transpose(-1, -2))       # 除以根号hidden_size
        return logits


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([2, 3, 4])
    print(a + b)