"""
Dataloaders for VCR
"""
import json
import os

import numpy as np
import torch
from allennlp.data.dataset import Batch    # ALLENNLP0.8独有
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR, IMAGE_DESIRED_HEIGHT, IMAGE_DESIRED_WIDTH
import time
import codecs
from tqdm import tqdm

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']


class VCRTokenizer:
    """Turn a detection list into what we want: some text, as well as some tags."""
    def __init__(self, obj2ind, obj2type, pad_ind, add_image_as_box=True) -> None:
        '''
        :param obj2ind: Mapping of the old ID -> new ID (which will be used as the tag)
        :param obj2type: [person, person, pottedplant] indexed by the old labels
        :param pad_ind: padding value
        :param add_image_as_box:
        '''
        self.obj2ind = obj2ind
        self.obj2type = obj2type
        self.pad_ind = pad_ind
        # 第一个object的index是2或者1
        self.add_image_as_box = add_image_as_box
        self.pad_ind = 0
        # self.pad_ind = 1 if self.add_image_as_box else 0
        self.obj_start_ind = 2 if self.add_image_as_box else 1

    def __call__(self, tokens, embeddings):
        """ tokens: Tokenized sentence with detections collapsed to a list. """
        new_tokens_with_tags = []
        for tok in tokens:
            if isinstance(tok, list):
                for int_name in tok:
                    obj_type = self.obj2type[int_name]
                    new_ind = self.obj2ind[int_name]
                    assert new_ind >= 0
                    # 与R2C保持一致
                    if obj_type == 'person':
                        text2use = GENDER_NEUTRAL_NAMES[(new_ind - self.obj_start_ind) % len(GENDER_NEUTRAL_NAMES)]
                    else:
                        text2use = obj_type
                    new_tokens_with_tags.append((text2use, new_ind))
            else:
                new_tokens_with_tags.append((tok, self.pad_ind))

        # 记得回来删
        assert len(new_tokens_with_tags) == embeddings.shape[0]
        text_field = BertField([Token(x[0]) for x in new_tokens_with_tags], embeddings, padding_value=0)
        tags = SequenceLabelField([x[1] for x in new_tokens_with_tags], text_field)
        return text_field, tags


class VCR(Dataset):
    def __init__(self, split, only_use_relevant_dets=True, add_image_as_a_box=True, embs_to_load='bert_da',
                 conditioned_answer_choice=0):
        """

        :param split: train, val, or test
        :param mode: answer or rationale
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the question and answer.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects.
        :param embs_to_load: Which precomputed embeddings to load.
        :param conditioned_answer_choice: If you're in test mode, the answer labels aren't provided, which could be
                                          a problem for the QA->R task. Pass in 'conditioned_answer_choice=i'
                                          to always condition on the i-th answer.       这是啥意思？？？？？？？？？？怎么test的时候还有这种操作？？？？？？？？？解释→  https://groups.google.com/forum/?hl=en#!topic/visualcommonsense/lxEgFYRz5ho
        """
        if split not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format('answer-rationale'))
        print("Loading {} embeddings".format(split), flush=True)
        self.split = split
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        self.add_image_as_a_box = add_image_as_a_box
        self.conditioned_answer_choice = conditioned_answer_choice

        with open(os.path.join(VCR_ANNOTS_DIR, split, '{}.jsonl'.format(split)), 'r') as f:
            self.items = np.array(list(f))

        self.token_indexers = {'elmo': ELMoTokenCharactersIndexer()}
        self.vocab = Vocabulary()

        with open(os.path.join(VCR_ANNOTS_DIR, 'dataloaders', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]       # 这里提到了background，思考一下以后如何利用background
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        self.embs_to_load = embs_to_load
        self.h5fn_answer = os.path.join(VCR_ANNOTS_DIR, self.split, f'{self.embs_to_load}_answer_{self.split}.h5')
        self.h5fn_rationale = os.path.join(VCR_ANNOTS_DIR, self.split, f'{self.embs_to_load}_rationale_{self.split}.h5')
        self.h5fn_image = os.path.join(VCR_ANNOTS_DIR, self.split, f'attribute_features_{self.split}.h5')


    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['mode', 'split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")
        ###############################这个地方在测试的时候要注意修改！！！！！！！！！！！！！######################################
        stuff_to_return = [
            cls(split='test', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item):
        """
        We might want to use fewer detections so lets do so.
        :param item:
        :param question:
        :param answer_choices:
        :return dets2use: index of dets to use selected from all the objects in each image
        :return old_det_to_new_ind: give each objects an new index,
            -1 indicates invalid 0-th indicates whole image if self.add_image_as_a_box
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['answer_choices'] + item['rationale_choices']     # test时需要修改

        # 过滤掉不相关的object，提示：测试only use relevant detection！！
        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            # select all people if there are no objects mentioned in answers and questions
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 2
        else:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        item = json.loads(self.items[index])
        instance_dict = {}
        dets2use, old_det_to_new_ind = self._get_dets_to_use(item)
        vcr_tokenizer = VCRTokenizer(old_det_to_new_ind, item['objects'], self.add_image_as_a_box)

        ######################################以下是Q2A的数据处理部分##################################################

        with h5py.File(self.h5fn_answer, 'r') as h5:
            grp_items_answer = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}            # (n, 768) dict_keys(['answer_answer0', 'answer_answer1', 'answer_answer2', 'answer_answer3', 'ctx_answer0', 'ctx_answer1', 'ctx_answer2', 'ctx_answer3'])        ['answer_rationale0', 'answer_rationale1', 'answer_rationale2', 'answer_rationale3', 'ctx_rationale0', 'ctx_rationale1', 'ctx_rationale2', 'ctx_rationale3']

        if 'endingonly' not in self.embs_to_load:
            questions_answer_tokenized, question_answer_tags = zip(*[vcr_tokenizer(
                item['question'],
                grp_items_answer[f'ctx_answer{i}']
            ) for i in range(4)])
            instance_dict['question_answer'] = ListField(list(questions_answer_tokenized))
            instance_dict['question_answer_tags'] = ListField(list(question_answer_tags))

        answers_tokenized, answer_tags = zip(*[vcr_tokenizer(
            answer,
            grp_items_answer[f'answer_answer{i}']
        ) for i, answer in enumerate(item['answer_choices'])])

        instance_dict['answers'] = ListField(list(answers_tokenized))
        instance_dict['answer_tags'] = ListField(list(answer_tags))


        ######################################以下是QA2R的数据处理部分################################################
        with h5py.File(self.h5fn_rationale, 'r') as h5_rationale:
            grp_items_rationale = {k: np.array(v, dtype=np.float16) for k, v in h5_rationale[str(index)].items()}

        condition_key = self.conditioned_answer_choice if self.split == "test" else ""
        conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
        question_rationale = item['question'] + item['answer_choices'][conditioned_label]

        if 'endingonly' not in self.embs_to_load:
            questions_rationale_tokenized, question_rationale_tags = zip(*[vcr_tokenizer(
                question_rationale,
                grp_items_rationale[f'ctx_rationale{condition_key}{i}']
            ) for i in range(4)])
            instance_dict['question_rationale'] = ListField(list(questions_rationale_tokenized))
            instance_dict['question_rationale_tags'] = ListField(list(question_rationale_tags))

        rationale_tokenized, rationale_tags = zip(*[vcr_tokenizer(
            rationale,
            grp_items_rationale[f'answer_rationale{condition_key}{i}']
        ) for i, rationale in enumerate(item['rationale_choices'])])

        instance_dict['rationales'] = ListField(list(rationale_tokenized))
        instance_dict['rationale_tags'] = ListField(list(rationale_tags))

        ####################################各种metadata数据处理部分##################################################
        if self.split != 'test':
            instance_dict['answer_label'] = LabelField(item['answer_label'], skip_indexing=True)
            instance_dict['rationale_label'] = LabelField(item['rationale_label'], skip_indexing=True)
        # instance_dict['metadata'] = MetadataField({'annot_id': item['annot_id'], 'ind': index, 'movie': item['movie'],
        #                                            'img_fn': item['img_fn'],
        #                                            'question_number': item['question_number']})

        ##########################################图片处理部分########################################################
        with h5py.File(self.h5fn_image, 'r') as h5_features:
            # pytoch1.1
            img_id = item['img_id'].split('-')[-1]
            group_image = {k: np.array(v) for k, v in h5_features[img_id].items()}
            image_feature = group_image['features'][[0]+(dets2use+1).tolist()]
            tag_boxes = group_image['boxes']
        zeros = np.zeros((1,2048), dtype=np.float32)
        if self.add_image_as_a_box:
            image_feature = np.concatenate((zeros, image_feature), axis=0)
        else:
            image_feature = np.concatenate((zeros, image_feature[1:]), axis=0)
        instance_dict['image_features'] = ArrayField(image_feature, padding_value=0)

        ###################################################################
        # Load boxes.
        with open(os.path.join(VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((boxes[0], boxes))
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels
        # 第一个object是0
        boxes = np.row_stack((boxes[0], boxes))
        obj_labels = [81] + obj_labels

        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)
        instance_dict['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])
        assert np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2]))

        instance = Instance(instance_dict)
        instance.index_fields(self.vocab)
        return instance


def collate_fn(data, to_gpu=False):
    """Creates mini-batch tensors
    """
    instances = data

    batch = Batch(instances)
    td = batch.as_tensor_dict()

    # ALLENNLP1.2
    if 'question_answer' in td:
        # 核对完model后再修改
        td['question_answer_mask'] = get_text_field_mask(td['question_answer'], num_wrapping_dims=1)
    if 'question_rationale' in td:
        td['question_rationale_mask'] = get_text_field_mask(td['question_rationale'], num_wrapping_dims=1)

    td['answer_mask'] = get_text_field_mask(td['answers'], num_wrapping_dims=1)
    td['rationale_mask'] = get_text_field_mask(td['rationales'], num_wrapping_dims=1)

    # pytorch1.1
    td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
    td['box_mask'][:, :1] = 0
    td['objects'] = torch.where(td['objects']==-1, torch.full(td['objects'].shape, 81, dtype=torch.int64), td['objects'])
    return td


class VCRLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, **kwargs):
        loader = cls(
            dataset=data,
            shuffle=data.is_train,
            collate_fn=lambda x: collate_fn(x, to_gpu=False),
            drop_last=data.is_train,
            **kwargs,
        )
        return loader

# You could use this for debugging maybe
if __name__ == '__main__':
    from tqdm import tqdm
    torch.set_printoptions(threshold=500000000, linewidth=8000000)
    train, val, test = VCR.splits(only_use_relevant_dets=False)
    train_loader = VCRLoader.from_dataset(train, num_workers=16, batch_size=128)
    for _ in tqdm(train_loader):
        pass
    # for i in tqdm(range(len(train))):
    #     res = train[i]
#         print("done with {}".format(i))
