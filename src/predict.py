import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from transformers import BertForQuestionAnswering, BertTokenizer
from argparse import ArgumentParser
import torch
import json
from tqdm import tqdm
from typing import Optional, List, Any, Tuple
from torch.utils.data import DataLoader

from src.train import MODEL_PATH, DATA_PATH, PROJECT_PATH
from src.dataset.cmrc2018.dataloader import CMRC2018
from src.metrics.metrics import Metrics


class Predicter:
    def __init__(self, args):
        self.args = args
        self.model = BertForQuestionAnswering.from_pretrained(self.args.model_path).to(self.args.device)
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.dataset = CMRC2018(args=args, tokenizer=self.tokenizer)()
        self.validation_dataloader = DataLoader(self.dataset['validation'],
                                                batch_size=self.args.batch_size,
                                                num_workers=self.args.num_workers)
        pass

    def _find_best_answer(self,
                          input_ids: Optional[torch.FloatTensor],
                          start_logits: Optional[torch.FloatTensor],
                          end_logits: Optional[torch.FloatTensor],
                          tokenizer: BertTokenizer) -> List[str]:
        start_ids = torch.argmax(start_logits, dim=-1)
        end_ids = torch.argmax(end_logits, dim=-1)
        return [''.join(tokenizer.convert_ids_to_tokens(input_id[start:end].numpy()))
                for input_id, start, end in zip(input_ids, start_ids, end_ids)]

    def get_references(self) -> Tuple[List[str], List[List[str]]]:
        with open(self.args.validation_file, encoding="utf-8") as f:
            data = json.load(f)
            questions, answers = [], []
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        id_ = qa["id"]
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers.append([answer["text"].strip() for answer in qa["answers"]])
                        questions.append(qa["question"].strip())
        return questions, answers

    def to_json(self, obj: Any, save_file: str):
        with open(save_file, 'w', encoding='utf-8') as fp:
            json.dump(obj=obj, fp=fp, indent=4, ensure_ascii=False)

    def __call__(self) -> List[str]:
        answers = []
        self.model.eval()
        for inputs_dict in tqdm(self.validation_dataloader):
            inputs = {key: value.to(self.args.device) for key, value in inputs_dict.items()
                      if key in ['input_ids', 'token_type_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits, end_logits = outputs[0], outputs[1]
            answer = self._find_best_answer(input_ids=inputs_dict['input_ids'],
                                            start_logits=start_logits,
                                            end_logits=end_logits,
                                            tokenizer=self.tokenizer)
            answers.extend(answer)
        return answers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--max_length", type=int, default=512, help="Max length of input sentence")
    parser.add_argument("--batch_size", type=int, default=32, help="Max length of input sentence")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path of the dataset.")
    parser.add_argument("--predict_path",
                        type=str,
                        default=os.path.join(DATA_PATH, "predict"),
                        help="Path of the dataset.")
    parser.add_argument("--data_script",
                        type=str,
                        default=os.path.join(PROJECT_PATH, 'src/dataset/cmrc2018/dataloader.py'),
                        help="Path of the dataset.")
    parser.add_argument("--train_file",
                        type=str,
                        default=os.path.join(DATA_PATH, 'cmrc2018/train_test.json'),
                        help="Path of the dataset.")
    parser.add_argument("--validation_file",
                        type=str,
                        default=os.path.join(DATA_PATH, 'cmrc2018/dev.json'),
                        help="Path of the dataset.")
    parser.add_argument("--model_path",
                        type=str,
                        # default=os.path.join(MODEL_PATH, 'based-on-roberta-wwm-ext/checkpoint-1000'),
                        default=os.path.join(MODEL_PATH, 'based-on-roberta-wwm-ext-my/checkpoint-1000'),
                        help="Path of the project.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of subprocesses for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Predict
    predicter = Predicter(args=args)
    predicts = predicter()
    questions, answers = predicter.get_references()
    res = [("Question: " + item[0], "Answer:  " + item[1][0], "Predict: " + item[2]) for item in zip(questions, answers, predicts)]
    predicter.to_json(res, os.path.join(args.predict_path, "predicts_result.json"))

    # Metrics
    metrics = Metrics()
    f1, em = metrics.compute(references=answers, predictions=predicts)
    metrics_result = f"F1: {round(f1, 3)}, EM: {round(em, 3)}"
    metrics.to_json(metrics_result, os.path.join(args.predict_path, "metrics_result.json"))
    pass
