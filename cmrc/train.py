import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
from transformers import BertForQuestionAnswering, BertTokenizer, Trainer, TrainingArguments
from argparse import ArgumentParser
import torch

PROJECT_PATH = os.path.dirname(os.getcwd())  # get current working directory
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')
LOG_PATH = os.path.join(PROJECT_PATH, 'log')
# PRETRAINED_MODEL_PATH = 'hfl/chinese-roberta-wwm-ext'
PRETRAINED_MODEL_PATH = '/home/yckj2939/project/yckj_project/KBQA/qag/cmrc/pretrain/roberta/out_cmrc2018'
sys.path.append(PROJECT_PATH)
from cmrc.dataset.cmrc2018.dataloader import CMRC2018


def train(args):
    model = BertForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer))
    datasets = CMRC2018(args=args, tokenizer=tokenizer)()
    training_args = TrainingArguments(
        output_dir=args.model_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        remove_unused_columns=False,
        logging_dir=args.log_path,
        num_train_epochs=args.n_epochs,
        dataloader_num_workers=args.num_workers,
        evaluation_strategy='epoch'
    )
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=datasets['train'],
                      eval_dataset=datasets['validation'])
    trainer.train()
    trainer.save_model()
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path of the dataset.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path of the model.")
    parser.add_argument("--log_path", type=str, default=LOG_PATH, help="Path of the log.")
    parser.add_argument("--data_script", type=str, default=os.path.join(PROJECT_PATH, 'cmrc/dataset/cmrc2018/dataloader.py'), help="Path of the dataset.")
    parser.add_argument("--train_file", type=str, default=os.path.join(DATA_PATH, 'cmrc2018/train.json'), help="Path of the dataset.")
    parser.add_argument("--validation_file", type=str, default=os.path.join(DATA_PATH, 'cmrc2018/dev.json'), help="Path of the dataset.")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of input sentence")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses for data loading")
    parser.add_argument("--warmup_steps", type=int, default=500, help="The steps of warm up.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()
    train(args=args)
