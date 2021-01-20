"""
Evaluation script for CMRC 2018
version: v5 - special
Note:
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
"""
from typing import Tuple, List, Any
import re
import json
import nltk


class Metrics:
	def __init__(self):
		pass

	# split Chinese with English
	def _mixed_segmentation(self, in_str, rm_punc=False):
		in_str = in_str.lower().strip()
		segs_out = []
		temp_str = ""
		sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；', '’',
				   '《', '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』']
		for char in in_str:
			if rm_punc and char in sp_char:
				continue
			if re.search(u'[\u4e00-\u9fa5]', char) or char in sp_char:  # chinese utf-8 code: u4e00 - u9fa5
				if temp_str != "":
					ss = nltk.word_tokenize(temp_str)
					segs_out.extend(ss)
					temp_str = ""
				segs_out.append(char)
			else:
				temp_str += char
		# handling last part
		if temp_str != "":
			ss = nltk.word_tokenize(temp_str)
			segs_out.extend(ss)
		return segs_out

	# remove punctuation
	def _remove_punctuation(self, in_str):
		in_str = in_str.lower().strip()
		sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；', '’',
				   '《', '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』']
		out_segs = []
		for char in in_str:
			if char in sp_char:
				continue
			else:
				out_segs.append(char)
		return ''.join(out_segs)

	def _find_lcs(self, s1, s2):
		m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
		mmax = 0
		p = 0
		for i in range(len(s1)):
			for j in range(len(s2)):
				if s1[i] == s2[j]:
					m[i + 1][j + 1] = m[i][j] + 1
					if m[i + 1][j + 1] > mmax:
						mmax = m[i + 1][j + 1]
						p = i + 1
		return s1[p - mmax:p], mmax

	def _compute_em_score(self, references: List[str], prediction: str) -> float:
		em = 0
		prediction = self._remove_punctuation(prediction)
		for reference in references:
			reference = self._remove_punctuation(reference)
			if reference == prediction:
				em = 1
				break
		return em

	def _compute_f1_score(self, references: List[str], prediction: str) -> float:
		f1_scores = []
		prediction_segment = self._mixed_segmentation(prediction, rm_punc=True)
		for reference in references:
			reference_segment = self._mixed_segmentation(reference, rm_punc=True)
			lcs, lcs_len = self._find_lcs(reference_segment, prediction_segment)
			if lcs_len == 0:
				f1_scores.append(0)
				continue
			precision = 1.0 * lcs_len / len(prediction_segment)
			recall = 1.0 * lcs_len / len(reference_segment)
			f1 = (2 * precision * recall) / (precision + recall)
			f1_scores.append(f1)
		return max(f1_scores)

	def compute(self, *args, **kwargs) -> Tuple[float, float]:
		"""Compute the metrics.
		Args:
			We disallow the usage of positional arguments to prevent mistakes
			`predictions` (Optional list/array/tensor): predictions
			`references` (Optional list/array/tensor): references
			`**kwargs` (Optional other kwargs): will be forwared to the metrics
		Return:
			Dictionnary with the metrics if this metric is run on the main process (process_id == 0)
			None if the metric is not run on the main process (process_id != 0)
		"""
		if args:
			raise ValueError("Please call `compute` using keyword arguments.")
		predictions = kwargs.pop("predictions", None)
		references = kwargs.pop("references", None)
		f1, em, total_count = 0, 0, 0
		for reference_list, prediction in zip(references, predictions):
			total_count += 1
			f1 += self._compute_f1_score(reference_list, prediction)
			em += self._compute_em_score(reference_list, prediction)
		f1_score = 100.0 * f1 / total_count
		em_score = 100.0 * em / total_count
		return f1_score, em_score

	def to_json(self, obj: Any, save_file: str):
		with open(save_file, 'w', encoding='utf-8') as fp:
			json.dump(obj=obj, fp=fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
	res = json.load(open('../../data/predict/evaluate_results_9000.json', 'r', encoding='utf-8'))
	answers, predicts = [], []
	for item in res:
		answers.append(item["target"])
		predicts.append(item["result"])
	metrics = Metrics()
	f1, em = metrics.compute(references=answers, predictions=predicts)
	metrics_result = f"F1: {round(f1, 3)}, EM: {round(em, 3)}"
	print(metrics_result)
	pass