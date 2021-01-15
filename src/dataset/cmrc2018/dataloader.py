from transformers import BertTokenizer
from torch.utils.data import DataLoader
import os
import datasets
import json


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

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(cmrc2018): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        yield id_, {
                            "context": context,
                            "question": question,
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
        pass

    def _get_answer_span(self, context, answer):
        """return [start, end)"""
        length = len(answer)
        for i in range(len(context)):
            if context[i:i + length] == answer:
                return i, i + length
        return self.args.max_length, self.args.max_length  # No answer for ignore due to truncation.

    def _encode(self, instance):
        inputs_dict = self.tokenizer(text=instance['context'],
                                     text_pair=instance['question'],
                                     truncation='only_first',
                                     padding='max_length',
                                     max_length=self.args.max_length)
        inputs_dict['answer_ids'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instance['answers']['text'][0]))
        inputs_dict['start_positions'], inputs_dict['end_positions'] = self._get_answer_span(inputs_dict['input_ids'], inputs_dict['answer_ids'])
        # inputs_dict['input_tokens'] = self.tokenizer.convert_ids_to_tokens(inputs_dict['input_ids'])
        # inputs_dict['start_positions'] = instance['answers']['answer_start'][0]
        return inputs_dict

    def __call__(self):
        self.datasets = self.datasets.map(function=self._encode)
        columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
        self.datasets.set_format(type='torch', columns=columns)
        return self.datasets
