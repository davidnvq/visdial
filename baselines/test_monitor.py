import yaml
from visdial.metrics import Monitor
from visdial.data import VisDialDataset


config_yml = 'configs/lf_disc_faster_rcnn_x101.yml'
config = yaml.load(open(config_yml))

dialogs_jsonpath = '/Users/quanguet/datasets/visdial/data/visdial_1.0_val.json'
dense_annotations_jsonpath = '/Users/quanguet/datasets/visdial/data/visdial_1.0_val_dense_annotations.json'

dataset = VisDialDataset(config['dataset'], dialogs_jsonpath, dense_annotations_jsonpath)

monitor = Monitor(dataset, 'data/monitor.pkl')
img_ids = monitor.img_ids
img_dialogs = monitor.img_dialogs
elem = img_dialogs[img_ids[0]]


questions = []
answers = []

for img_id in img_ids:
	elem = img_dialogs[img_id]
	questions.append(elem['rel_question'])
	nonzero_rel_indices = elem['rel_scores'].nonzero().squeeze(-1)
	top_answers = [elem['rel_texts'][idx] for idx in nonzero_rel_indices]
	answers.append(top_answers)


template_questions = [
	'can you see',
	'can you tell',
	'is it',
	'is there',
	'are there',
	'is he',
	'is she',
	'does it',
	'does he',
	'does she',
	'do they',
	]

