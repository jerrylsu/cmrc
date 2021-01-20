from transformers import BertTokenizer
from torch.utils.data import DataLoader
import os
import datasets
import json
import re
import itertools
from typing import List


class Cmrc2018(datasets.GeneratorBasedBuilder):
    """TODO(cmrc2018): Short description of my dataset."""

    # TODO(cmrc2018): Set up version.
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(cmrc2018): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=None,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "highlight_sent": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(cmrc2018): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = {
            "train": self.config.data_files["train"],
            "validation": self.config.data_files["validation"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
        ]

    def _split_sentence(self, document: str, flag: str = "zh", limit: int = 510) -> List[str]:
        """
        Args:
            document:
            flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
            limit: 默认单句最大长度为510个字符

        Returns: Type:list
        """
        sent_list = []
        try:
            if flag == "zh":
                document = re.sub('(?P<quotation_mark>([。？！…](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 单字符断句符
                document = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'])', r'\g<quotation_mark>\n',
                                  document)  # 特殊引号
            elif flag == "en":
                document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 英文单字符断句符
                document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
            else:
                document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 单字符断句符
                document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                                  document)  # 特殊引号

            sent_list_ori = document.splitlines()
            for sent in sent_list_ori:
                # sent = sent.strip()
                if not sent:
                    continue
                else:
                    while len(sent) > limit:
                        temp = sent[0:limit]
                        sent_list.append(temp)
                        sent = sent[limit:]
                    sent_list.append(sent)
        except:
            sent_list.clear()
            sent_list.append(document)
        return sent_list

    def _get_split_sentences_positions(self, context):
        # split into sentences
        sents = self._split_sentence(context)
        # get positions of the sentences
        positions = []  # 闭区间
        for i, sent in enumerate(sents):
            if i == 0:
                start, end = 0, len(sent) - 1
            else:
                start, end = (prev_end), (prev_end + len(sent) - 1)
            prev_end = end + 1
            positions.append({'start': start, 'end': end})
        return positions, sents

    def _get_highlight_sentence(self, answer_start, context_sents, positions):
        for pos, sent in zip(positions, context_sents):
            if answer_start in range(pos["start"], pos["end"] + 1):
                return sent
        raise Exception("Not find highlight sentence.")

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(cmrc2018): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    positions, context_sents = self._get_split_sentences_positions(context)
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        highlight_sent = self._get_highlight_sentence(answer_starts[0], context_sents, positions)
                        yield id_, {
                            "context": context,
                            "question": question,
                            "highlight_sent": highlight_sent,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }


class CMRC2018:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.datasets = datasets.load_dataset(self.args.data_script,
                                              data_files={"train": self.args.train_file,
                                                          "validation": self.args.validation_file},
                                              cache_dir=os.path.join(self.args.data_path, 'cmrc2018'))
        self.all_keys = self._remove_duplicate(self.datasets)
        pass

    def _remove_duplicate(self, dataset):
        all_key = set()
        for data in dataset["train"]:
            all_key.add(data["highlight_sent"])
        for data in dataset["validation"]:
            all_key.add(data["highlight_sent"])
        return all_key

    def _data_filter(self, instance):
        if instance["highlight_sent"] in self.all_keys:
            self.all_keys.remove(instance["highlight_sent"])
            return True
        return False

    def _get_answer_span(self, context, answer):
        """return [start, end)"""
        length = len(answer)
        for i in range(len(context)):
            if context[i:i + length] == answer:
                return i, i + length
        return self.args.max_length, self.args.max_length  # No answer for ignore due to truncation.

    def _encode(self, instance):
        inputs_dict = self.tokenizer(text=instance['context'],
                                     text_pair=instance['highlight_sent'],
                                     truncation='only_first',
                                     padding='max_length',
                                     max_length=self.args.max_length)
        inputs_dict['question'] = instance["question"]
        inputs_dict['answer_ids'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instance['answers']['text'][0]))
        inputs_dict['start_positions'], inputs_dict['end_positions'] = self._get_answer_span(inputs_dict['input_ids'], inputs_dict['answer_ids'])
        # inputs_dict['input_tokens'] = self.tokenizer.convert_ids_to_tokens(inputs_dict['input_ids'])
        # inputs_dict['start_positions'] = instance['answers']['answer_start'][0]
        return inputs_dict

    def __call__(self):
        self.datasets = self.datasets.filter(function=self._data_filter)
        self.datasets = self.datasets.map(function=self._encode)
        #columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'answers', 'question']
        #self.datasets.set_format(type='torch', columns=columns)
        return self.datasets
