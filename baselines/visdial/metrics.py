"""
A Metric observes output of certain model, for example, in form of logits or
scores, and accumulates a particular metric with reference to some provided
targets. In context of VisDial, we use Recall (@ 1, 5, 10), Mean Rank, Mean
Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

Each ``Metric`` must atleast implement three methods:
    - ``observe``, update accumulated metric with currently observed outputs
      and targets.
    - ``retrieve`` to return the accumulated metric., an optionally reset
      internally accumulated metric (this is commonly done between two epochs
      after validation).
    - ``reset`` to explicitly reset the internally accumulated metric.

Caveat, if you wish to implement your own class of Metric, make sure you call
``detach`` on output tensors (like logits), else it will cause memory leaks.

Quang modified
==============
TODONE: A bug for batch_size = 1
```
# shape: (batch_size, num_options)
predicted_ranks = predicted_ranks.squeeze()

# should be changed to
predicted_ranks = predicted_ranks.squeeze(1)
"""
import torch
import pickle


def scores_to_ranks(scores: torch.Tensor):
	"""Convert model output scores into ranks."""
	batch_size, num_rounds, num_options = scores.size()
	scores = scores.view(-1, num_options)

	# sort in descending order - largest score gets highest rank
	sorted_ranks, ranked_idx = scores.sort(1, descending=True)

	# i-th position in ranked_idx specifies which score shall take this
	# position but we want i-th position to have rank of score at that
	# position, do this conversion
	ranks = ranked_idx.clone().fill_(0)
	for i in range(ranked_idx.size(0)):
		for j in range(num_options):
			ranks[i][ranked_idx[i][j]] = j
	# convert from 0-99 ranks to 1-100 ranks
	ranks += 1
	ranks = ranks.view(batch_size, num_rounds, num_options)
	return ranks


class SparseGTMetrics(object):
	"""
	A class to accumulate all metrics with sparse ground truth annotations.
	These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
	"""

	def __init__(self):
		self._rank_list = []

	def observe(
			self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
			):
		predicted_scores = predicted_scores.detach()

		# shape: (batch_size, num_rounds, num_options)
		predicted_ranks = scores_to_ranks(predicted_scores)
		batch_size, num_rounds, num_options = predicted_ranks.size()

		# collapse batch dimension
		predicted_ranks = predicted_ranks.view(
				batch_size * num_rounds, num_options
				)

		# shape: (batch_size * num_rounds, )
		target_ranks = target_ranks.view(batch_size * num_rounds).long()

		# shape: (batch_size * num_rounds, )
		predicted_gt_ranks = predicted_ranks[
			torch.arange(batch_size * num_rounds), target_ranks
		]
		self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))

	def retrieve(self, reset: bool = True):
		num_examples = len(self._rank_list)
		if num_examples > 0:
			# convert to numpy array for easy calculation.
			__rank_list = torch.tensor(self._rank_list).float()
			metrics = {
				"r@1" : torch.mean((__rank_list <= 1).float()).item(),
				"r@5" : torch.mean((__rank_list <= 5).float()).item(),
				"r@10": torch.mean((__rank_list <= 10).float()).item(),
				"mean": torch.mean(__rank_list).item(),
				"mrr" : torch.mean(__rank_list.reciprocal()).item(),
				}
		else:
			metrics = {}

		if reset:
			self.reset()
		return metrics

	def reset(self):
		self._rank_list = []


class NDCG(object):
	def __init__(self):
		self._ndcg_numerator = 0.0
		self._ndcg_denominator = 0.0

	def observe(
			self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
			):
		"""
		Observe model output scores and target ground truth relevance and
		accumulate NDCG metric.

		Parameters
		----------
		predicted_scores: torch.Tensor
			A tensor of shape (batch_size, num_options), because dense
			annotations are available for 1 randomly picked round out of 10.
		target_relevance: torch.Tensor
			A tensor of shape same as predicted scores, indicating ground truth
			relevance of each answer option for a particular round.
		"""
		predicted_scores = predicted_scores.detach()

		# shape: (batch_size, 1, num_options)
		predicted_scores = predicted_scores.unsqueeze(1)
		predicted_ranks = scores_to_ranks(predicted_scores)

		# shape: (batch_size, num_options)
		predicted_ranks = predicted_ranks.squeeze(1)
		batch_size, num_options = predicted_ranks.size()

		k = torch.sum(target_relevance != 0, dim=-1)

		# shape: (batch_size, num_options)
		_, rankings = torch.sort(predicted_ranks, dim=-1)
		# Sort relevance in descending order so highest relevance gets top rnk.
		_, best_rankings = torch.sort(
				target_relevance, dim=-1, descending=True
				)

		# shape: (batch_size, )
		batch_ndcg = []
		for batch_index in range(batch_size):
			num_relevant = k[batch_index]
			dcg = self._dcg(
					rankings[batch_index][:num_relevant],
					target_relevance[batch_index],
					)
			best_dcg = self._dcg(
					best_rankings[batch_index][:num_relevant],
					target_relevance[batch_index],
					)
			batch_ndcg.append(dcg / best_dcg)

		self._ndcg_denominator += batch_size
		self._ndcg_numerator += sum(batch_ndcg)

	def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
		sorted_relevance = relevance[rankings].cpu().float()
		discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
		return torch.sum(sorted_relevance / discounts, dim=-1)

	def retrieve(self, reset: bool = True):
		if self._ndcg_denominator > 0:
			metrics = {
				"ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
				}
		else:
			metrics = {}

		if reset:
			self.reset()
		return metrics

	def reset(self):
		self._ndcg_numerator = 0.0
		self._ndcg_denominator = 0.0


class Monitor(object):

	def __init__(self, dataset, save_path='data/monitor_val.pkl'):
		self.dataset = dataset
		self.save_path = save_path
		self.img_dialogs = {}
		self.img_ids = []
		self.init_dialogs()

	def get_elem_dict(self, elem):

		dialog = elem['dialog']
		questions = [' '.join(r['question']) for r in dialog]
		answers = [' '.join(r['answer']) for r in dialog]
		opts = [[' '.join(r['answer_options'][i])
		         for i in range(100)] for r in dialog]

		scores = [[] for _ in range(10)]
		attn_weights = [[] for _ in range(10)]

		rel_ans_idx = elem['rel_ans_idx']

		round_id = elem['round_id'] - 1
		rel_scores, rel_indices = torch.sort(elem['gt_relevance'], descending=True)
		rel_scores = rel_scores[:10]
		rel_indices = rel_indices[:10]

		if rel_ans_idx not in rel_indices:
			rel_indices = torch.cat([rel_indices, torch.tensor([rel_ans_idx])], dim=-1)
			rel_scores = torch.cat([rel_scores, rel_scores[-1:]], dim=-1)

		rel_texts = [' '.join(dialog[round_id]['answer_options'][idx])
		             for idx in rel_indices]

		rel_question = ' '.join(dialog[round_id]['question'])
		rel_answer = ' '.join(dialog[round_id]['answer'])

		return {
			'caption'     : elem['caption'],
			'dialog'      : [questions, answers],
			'opts'        : opts,
			'scores'      : scores,
			'rel_answer'  : rel_answer,
			'rel_indices' : rel_indices,
			'rel_scores'  : rel_scores,
			'round_id'    : elem['round_id'],
			'rel_texts'   : rel_texts,
			'rel_preds'   : [],
			'rel_question': rel_question,
			'rel_ans_idx': rel_ans_idx,
			'attn_weights': attn_weights,
			}

	def update_elem(self, img_dialogs, img_id, output, target, attn_weight):
		# shape: output [10, 100]
		# shape: target [10, ]
		elem_dict = img_dialogs[img_id]

		# Ground Truth Relevance in round_id only
		round_id = elem_dict['round_id']

		# shape: [1, ] # 1 round - ground truth
		gtr_pred = output[round_id - 1][elem_dict['gtr_idx']].cpu().numpy()
		elem_dict['gtr_preds'].append(gtr_pred)

		# All answers in 10 rounds
		for r in range(10):
			prob100_dict = {ans_opt: value.item()
			                for ans_opt, value in
			                zip(elem_dict['opts'][r], output[r])}

			target_key = elem_dict['opts'][r][target[r]]

			sorted_keys = sorted(prob100_dict,
			                     key=prob100_dict.get,
			                     reverse=True)

			if target_key not in sorted_keys:
				sorted_keys.append(target_key)

			prob10_dict = {key: prob100_dict[key]
			               for key in sorted_keys}

			elem_dict['scores'][r].append(prob10_dict)  # r = round

			if attn_weight is not None:
				elem_dict['attn_weights'][r].append(attn_weight[r].cpu().numpy())

	def init_dialogs(self):
		for i in range(len(self.dataset)):
			elem = self.dataset.__getitem__(i, monitor=True)
			self.img_ids.append(elem['img_id'])
			self.img_dialogs[elem['img_id']] = self.get_elem_dict(elem)


	def update(self, batch_img_ids, batch_logits, batch_targets, attn_weights=None):

		batch_output = torch.softmax(batch_logits, dim=-1)
		if attn_weights is None:
			attn_weights = [None] * len(batch_logits)

		for i in range(len(batch_img_ids)):
			self.update_elem(self.img_dialogs,
			                 batch_img_ids[i].item(),
			                 batch_output[i],
			                 batch_targets[i],
			                 attn_weights[i])


	def export(self):
		with open(self.save_path, 'wb') as file:
			pickle.dump([self.img_ids, self.img_dialogs], file)


def test_monitor():
	monitor = Monitor()
	pass
