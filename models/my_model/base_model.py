import torch
from torch import nn, Tensor
from torch.nn import functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, InputVariationalDropout, TimeDistributed
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from functools import reduce
from operator import mul
from typing import Dict, Any
from models.my_model.fc import *
import torch
import math


@Model.register('my_model')
class MyModel(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            span_encoder_q: Seq2SeqEncoder,
            span_encoder_a: Seq2SeqEncoder,
            detector: Model = None,
            semantic: bool = True,
            rnn_input_dropout: float = 0.3,
            output_dropout: float = 0.3,
            initializer: InitializerApplicator = InitializerApplicator()
    ) -> None:
        super().__init__(vocab)
        self.object_embed = torch.nn.Embedding(num_embeddings=82, embedding_dim=128, padding_idx=81) if semantic else None
        if rnn_input_dropout > 0:
            self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(rnn_input_dropout))
        else:
            self.rnn_input_dropout = None
        self.obj_downsample = FCNet(input_dim=2176, output_dim=512, input_dropout=0.1)
        self.span_encoder_q = span_encoder_q
        self.span_encoder_a = span_encoder_a
        self.attention = AttentionOnAttention()
        self.reasoning_mlp = torch.nn.Sequential(
            nn.Dropout(output_dropout),
            nn.Linear(512 * 2, 512),
            nn.LeakyReLU(negative_slope=0.1),
            # TimeDistributed(nn.BatchNorm1d(512)),
            nn.Dropout(output_dropout),
            nn.Linear(512, 1)
        )

        self.detector = detector

        self._q2a_loss = nn.CrossEntropyLoss()
        self._qa2r_loss = nn.CrossEntropyLoss()

        self._q2a_accuracy = CategoricalAccuracy()
        self._qa2r_accuracy = CategoricalAccuracy()

        initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(
            self,
            question,
            question_mask,
            question_tags,
            answers,
            answer_mask,
            answer_tags,
            object_reps,
            box_mask,
            encoder_q,
            encoder_a,
    ):
        if self.rnn_input_dropout:
            question = self.rnn_input_dropout(question['bert'])
            answers = self.rnn_input_dropout(answers['bert'])
        question_related_objs = self._collect_obj_reps(question_tags, object_reps)
        answer_related_objs = self._collect_obj_reps(answer_tags, object_reps)
        question = torch.cat((question, question_related_objs), dim=-1)
        answers = torch.cat((answers, answer_related_objs), dim=-1)

        q_seq_sizes = question.shape
        question = encoder_q(question.view(-1, *q_seq_sizes[-2:]), question_mask.view(-1, q_seq_sizes[-2]))
        q_rep = self.get_seq_last(question, question_mask, bidirectional=True)

        a_seq_sizes = answers.shape
        answers = encoder_a(
            answers.view(-1, *a_seq_sizes[-2:]),
            answer_mask.view(-1, a_seq_sizes[-2]),
            q_rep.repeat(2, 1, 1)
        )
        a_rep = self.get_seq_last(answers, answer_mask, bidirectional=True)

        question = question.view(*q_seq_sizes[:-1], -1)
        answers = answers.view(*a_seq_sizes[:-1], -1)
        q_rep = q_rep.view(*q_seq_sizes[:-2], -1)
        a_rep = a_rep.view(*a_seq_sizes[:-2], -1)

        span_rep = torch.cat([question, answers], dim=-2)
        span_mask = torch.cat([question_mask, answer_mask], dim=-1)
        return span_rep, span_mask, q_rep, a_rep

    def get_seq_last(self, seq: torch.Tensor, mask: torch.Tensor, bidirectional: bool = True):
        """add forward and backward representation if bidirectional in sequence"""
        leading_dims = seq.shape[:-2]
        seq_len, emd_dim = seq.shape[-2:]
        leading_size = reduce(mul, leading_dims)
        seq = seq.view(-1, seq_len, emd_dim)
        first_position = mask.new_zeros(leading_size).long()
        last_position = (mask.sum(-1) - 1).view(-1)
        if not bidirectional:
            return seq[torch.arange(leading_size), last_position].view(*leading_dims, emd_dim)
        seq_first_emd = seq[torch.arange(leading_size), first_position, emd_dim//2:]
        seq_last_emd = seq[torch.arange(leading_size), last_position, :emd_dim//2]
        # 返回forward与backward最后时间步的相加
        return (seq_last_emd + seq_first_emd).view(*leading_dims, emd_dim//2)

    def forward(
            self,
            image_features: torch.Tensor,
            objects: torch.LongTensor,
            boxes: torch.Tensor,
            box_mask: torch.LongTensor,
            question_answer: Dict[str, torch.Tensor],
            question_answer_tags: torch.LongTensor,
            question_answer_mask: torch.LongTensor,
            question_rationale: Dict[str, torch.Tensor],
            question_rationale_tags: torch.LongTensor,
            question_rationale_mask: torch.LongTensor,
            answers: Dict[str, torch.Tensor],
            answer_tags: torch.LongTensor,
            answer_mask: torch.LongTensor,
            rationales: Dict[str, torch.Tensor],
            rationale_tags: torch.LongTensor,
            rationale_mask: torch.LongTensor,
            # metadata: List[Dict[str, Any]] = None,
            answer_label: torch.LongTensor = None,
            rationale_label: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        max_len = int(box_mask.sum(1).max().item())
        #############################################################################################################
        for tag_type, the_tags in (('question_answer', question_answer_tags), ('answer', answer_tags),
                                   ('question_rationale', question_rationale_tags), ('rationale', rationale_tags)):
            if int(the_tags.max()) > (max_len):
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags))
        feats_to_downsample = image_features if self.object_embed is None else torch.cat(
            (image_features, self.object_embed(objects)), -1)

        obj_reps = self.obj_downsample(feats_to_downsample)
        obj_reps = obj_reps * box_mask[:, :, None].float()  # batch_size, max_obj_num, v_dim

        # ===================================== 对Q2A task进行计算 ================================================ #

        q2a_span_rep, q2a_span_mask, q2a_q_rep, q2a_a_rep = self.embed_span(
            question_answer, question_answer_mask, question_answer_tags,
            answers, answer_mask, answer_tags,
            obj_reps, box_mask,
            self.span_encoder_q, self.span_encoder_a
        )

        q2a_att_output = self.attention(
            span_rep=q2a_span_rep,
            span_mask=q2a_span_mask,
            q_rep=q2a_q_rep,
            a_rep=q2a_a_rep,
            obj_reps=obj_reps,
            box_mask=box_mask
        )

        q2a_logits = self.reasoning_mlp(q2a_att_output['reasoning_inp']).squeeze(-1)
        q2a_class_probabilities = F.softmax(q2a_logits, dim=-1)

        # ===================================== 对QA2R task进行计算 ================================================ #

        qa2r_span_rep, qa2r_span_mask, qa2r_q_rep, qa2r_a_rep = self.embed_span(
            question_rationale, question_rationale_mask, question_rationale_tags,
            rationales, rationale_mask, rationale_tags,
            obj_reps, box_mask,
            self.span_encoder_q, self.span_encoder_a
        )

        qa2r_att_output = self.attention(
            span_rep=qa2r_span_rep,
            span_mask=qa2r_span_mask,
            q_rep=qa2r_q_rep,
            a_rep=qa2r_a_rep,
            obj_reps=obj_reps,
            box_mask=box_mask
        )

        qa2r_logits = self.reasoning_mlp(qa2r_att_output['reasoning_inp']).squeeze(-1)
        qa2r_class_probabilities = F.softmax(qa2r_logits, dim=-1)
        # qa2r中attention的熵
        # ===================================== 输出 ================================================ #

        output_dict = {
            "q2a_label_logits": q2a_logits, "q2a_label_probs": q2a_class_probabilities,
            "qa2r_label_logits": qa2r_logits, "qa2r_label_probs": qa2r_class_probabilities,
        }
        if (answer_label is not None) and (rationale_label is not None):
            q2a_loss = self._q2a_loss(q2a_logits, answer_label.long().view(-1))
            self._q2a_accuracy(q2a_logits, answer_label)
            output_dict["q2a_loss"] = q2a_loss[None]

            qa2r_loss = self._qa2r_loss(qa2r_logits, rationale_label.long().view(-1))
            self._qa2r_accuracy(qa2r_logits, rationale_label)
            output_dict["qa2r_loss"] = qa2r_loss[None]

            loss = q2a_loss + qa2r_loss

            if self.detector is not None:
                align_output = self.detector(
                    # q2a_att_output['att1_weighted_logits'],
                    q2a_att_output,
                    # qa2r_att_output['att1_weighted_logits'],
                    qa2r_att_output,
                    answer_label,
                    rationale_label,
                    box_mask.unsqueeze(1)
                )
                for key, value in align_output.items():
                    output_dict[f'align_{key}'] = value

                if 'align_loss' in output_dict:
                    loss = loss + output_dict['align_loss']
            output_dict["loss"] = loss[None]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, Any]:
        metrics = {
            'q2a_accuracy': self._q2a_accuracy.get_metric(reset),
            'qa2r_accuracy': self._qa2r_accuracy.get_metric(reset),
        }
        return metrics


def masked_softmax(vector: Tensor, mask: Tensor = None, dim: int = None):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        huge_negative = torch.zeros_like(vector).masked_fill(mask == 0, -1e9)
        scores = vector + huge_negative
        result = F.softmax(scores, dim=-1)
    return result


class AttDP(nn.Module):
    def __init__(
            self,
            scaled: bool = True
    ):
        super().__init__()

        self.query_proj = FCNet(512, 512, input_dropout=None, output_dropout=0.3)
        self.key_proj = FCNet(512, 512, input_dropout=None, output_dropout=0.3)
        self.hidden_size = self.query_proj.output_dim
        self.scaled = scaled

    def forward(self, query, key, query_mask=None, key_mask=None):
        """shape of input and mask should be the same in first several dimensions"""
        logits = self.logits(query, key)
        if key_mask is not None:
            w = masked_softmax(logits, key_mask.unsqueeze(-2))
        else:
            w = torch.softmax(logits, dim=-1)
        if query_mask is not None:
            w = w * query_mask.unsqueeze(-1)  # mask padding in sentence
        return w

    def forward_with_logits(self, query, key, query_mask=None, key_mask=None):
        logits = self.logits(query, key)
        if key_mask is not None:
            w = masked_softmax(logits, key_mask.unsqueeze(-2))
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
        if self.scaled:
            logits = logits/math.sqrt(self.hidden_size)
        return logits


class AttentionOnAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention1 = AttDP(scaled=False)
        self.attention2 = AttDP(scaled=False)

    def forward(
            self,
            span_rep: Tensor,
            span_mask: Tensor,
            q_rep: Tensor,
            a_rep: Tensor,
            obj_reps: Tensor,
            box_mask: Tensor
    ):
        _, (att1, att1_logits) = self.attention1.forward_with_logits(
            span_rep, obj_reps.unsqueeze(1), query_mask=span_mask.float(), key_mask=box_mask)

        att2_query_ = torch.cat([q_rep, a_rep], dim=-1)
        att2_query = att2_query_ + obj_reps[:, [1]]

        span_joint_rep = att1.matmul(obj_reps.unsqueeze(1)) + span_rep
        att2_value, (att2, att2_logits) = self.attention2.forward_with_logits(
            att2_query.unsqueeze(-2), span_joint_rep, key_mask=span_mask)

        att_last = torch.matmul(att2, att1).squeeze(2)
        attended_o = torch.einsum('bno, bod->bnd', [att_last, obj_reps])
        reasoning_inp = torch.cat([att2_query_.squeeze(2), attended_o], dim=-1)

        att1_weighted_logits = att2.matmul(att1_logits).squeeze(2)
        output = {
            'att1': att1,
            'att1_logits': att1_logits,
            'att1_weighted_logits': att1_weighted_logits,
            'att2': att2,
            'att2_logits': att2_logits,
            'att_last': att_last,
            'reasoning_inp': reasoning_inp
        }
        return output


@Model.register('base_4_aligner')
class Base4Aligner(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            logits_predictor: Model = None,
            align_loss_model: Model = None,
            align_key: str = 'att_last',
    ) -> None:
        super().__init__(vocab)
        self.align_key = align_key
        self.logits_predictor = logits_predictor
        self.align_loss_model = align_loss_model
        self._align_accuracy = CategoricalAccuracy()

    def forward(
            self,
            q2a_att: Dict[str, Tensor],
            qa2r_att: Dict[str, Tensor],
            answer_label: Tensor,
            rationale_label: Tensor,
            box_mask: Tensor = None
    ):
        """bx4xn"""
        q2a_att = q2a_att[self.align_key]
        qa2r_att = qa2r_att[self.align_key]
        output = dict()
        if self.logits_predictor is None:
            return output
        q2a_true_att = q2a_att[torch.arange(q2a_att.shape[0]), answer_label].unsqueeze(1)
        qa2r_true_att = qa2r_att[torch.arange(qa2r_att.shape[0]), rationale_label].unsqueeze(1)
        align_q2a_logits = self.logits_predictor(qa2r_true_att, q2a_att, box_mask)
        align_qa2r_logits = self.logits_predictor(q2a_true_att, qa2r_att, box_mask)
        output['q2a_logits'] = align_q2a_logits
        output['qa2r_logits'] = align_qa2r_logits
        if (answer_label is not None) and (rationale_label is not None):
            if self.align_loss_model is not None:
                align_loss = self.align_loss_model(align_q2a_logits, answer_label) + \
                             self.align_loss_model(align_qa2r_logits, rationale_label)
                output['loss'] = align_loss
            self._align_accuracy(align_q2a_logits, answer_label)
        return output

    def get_metric(self, reset: bool = False):
        return self._align_accuracy.get_metric(reset=reset)


@Model.register('inner_product_predictor')
class InnerProductPredictor(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            norm: Model = None,
            logits_weight: float = 1.0,
    ):
        super().__init__(vocab)
        self.norm = norm
        self.logits_weight = logits_weight

    def forward(self, q2a_att: Tensor, qa2r_att: Tensor, box_mask: Tensor = None):
        if self.norm is not None:
            q2a_att = self.norm(q2a_att)
            qa2r_att = self.norm(qa2r_att)
        align_logits = self.logits_weight * q2a_att.matmul(qa2r_att.transpose(-1, -2)).view(q2a_att.shape[0], -1)
        return align_logits


@Model.register('align_cross_entropy_loss')
class AlignCrossEntropyLoss(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            weight: float,
            classes: int = 16,
            smoothing: float = 0
    ):
        super().__init__(vocab)
        self.weight = weight
        self.smoothing = smoothing
        self.classes = classes
        self._loss = nn.CrossEntropyLoss()

    def forward(self, align_logits: Tensor, align_gt: Tensor):
        with torch.no_grad():
            true_dist = align_logits.new_zeros(align_logits.shape) + self.smoothing / (self.classes - 1)
            true_dist.scatter_(1, align_gt.data.unsqueeze(1), 1 - self.smoothing)
        lsm = F.log_softmax(align_logits, dim=-1)
        align_loss = - (lsm * true_dist).sum(-1)
        align_loss = self.weight * align_loss.mean(-1)
        # align_loss = self.weight * self._loss(align_logits, align_gt.long().view(-1))
        return align_loss
