import os
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

val_path = '/home/quanguet/datasets/visdial/bottom-up/val2018_resnet101_faster_rcnn_genome_36.h5'
dir_path = '/home/quanguet/datasets/visdial/raw_images/VisualDialog_val2018'
pattern = 'VisualDialog_val2018_000000{}.jpg'

def img_id_to_path(img_id, dir_path, pattern='VisualDialog_val2018_000000{:06d}.jpg'):
	return os.path.join(dir_path, pattern.format(img_id))


def draw_box(image, box, color, thickness=2):
	'''
	Draw a box on an image with a give color.
	:param image: (PIL Image or np.array) The image to draw on
	:param box: (list) a list of 4 elements (xmin, ymin, xmax, ymax)
	:param color: The color of the box.
	:param thickness: The thickness of the lines to draw the box.
	:return: None
	'''
	b = np.array(box).astype(int)
	cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
	return image


def draw_boxes_with_scores(img, boxes, scores, topk=5, skip_indices=[]):
	"""
	Args:
		img (np.array) [H, W, 3]
		boxes (np.array) [num_boxes, 4] a box is represented by (x1, y1, x2, y2)
		scores (np.array) [num_boxes,]
		topk: top boxes with the highest scores.

	Return:
		new_img: (np.array)
	"""
	new_img = (img * 0.3).astype('uint8')

	# sorting
	selection = np.argsort(scores)[-topk:]
	alpha_scores = np.array([1.0, 0.5, 0.25, 0.125, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

	for al_idx, i in enumerate(selection):
		if i in skip_indices:
			continue
		score = alpha_scores[al_idx]
		box = boxes[i]
		# draw the region
		alpha = score * 1.2 if score * 1.2 < 1.0 else 1.0
		box = box.astype(int)
		x1, y1, x2, y2 = box
		new_img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * alpha).astype('uint8')
		cv2.rectangle(new_img, (x1, y1), (x2, y2), color=(int(score * 255), 0, 0), thickness=2)

		# Add text
		font_scale = 1.0
		font_face = cv2.FONT_HERSHEY_PLAIN
		text = "{}:{:1.2f}".format(len(boxes) - i, score)

		# get the width and height of the text box
		tw, th = cv2.getTextSize(text, font_face, font_scale, thickness=1)[0]

		# set the text start position
		tx1, ty1 = x1, y1
		tx2, ty2 = tx1 + tw - 2, ty1 - th - 2

		# draw the text box
		cv2.rectangle(new_img, (tx1, ty1), (tx2, ty2), (255,255,255), cv2.FILLED)

		# draw teh text
		cv2.putText(new_img, text, (tx1, ty1), font_face, font_scale, (0, 0, 0),thickness=1)
	return new_img


def draw_caption(image, box, caption):
	'''
	Draw a caption above the box in an image
	:param image: (PIL Image) The image to draw on.
	:param box:  (list) A list of 4 elements (xmin, ymin, xmax, ymax)
	:param caption: (str) The text to draw.
	:return: None
	'''
	b = np.array(box).astype(int)
	black_color = (0, 0, 0)
	white_color = (255, 255, 255)

	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, black_color, 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, white_color, 1)
	return image


def draw_boxes(image, boxes, color, thickness=2):
	'''
	Draw boxes on an image with a give color.
	:param image: (PIL Image) The image to draw on.
	:param boxes: (list) a list of boxes with shape [N, 4]
	:param color: (int) The color of the boxes
	:param thickness: (int) The thickness of the lines to draw boxes with.
	:return: None
	'''
	for box in boxes:
		draw_box(image, box, color, thickness=thickness)
	return image

def draw_detections(image, boxes, scores, labels, color=(255, 0, 0), label_to_name=None,
                    score_threshold=0.3, show_image=True):
	'''
	Draw detections in an image
	:param image: (PIL Image or np.array) The image to draw on.
	:param boxes: (list) The list of boxes of shape [N, 4]
	:param scores: (list) A list of N confidence scores.
	:param labels: (list) A list of N labels for each box.
	:param colors: (list) The colors of the boxes of different classes.
	:param label_to_name: (dictionary) Mapping a label to name
	:param score_threshold: (float) Threshold to determine which detections to draw.
	:param show_image: (boolean) If True, show the drawn image.
	:return:
	'''
	image = np.array(image)
	boxes = np.array(boxes)
	scores = np.array(scores)
	selection = np.where(scores > score_threshold)[0]

	for i in selection:
		c = color
		draw_box(image, boxes[i, :], color=c)
		# Draw labels and confidence scores
		conf_score_text = '%.2f' % (scores[i])
		label_text = label_to_name(labels[i]) if label_to_name else 'score'
		caption = label_text + ':' + conf_score_text
		draw_caption(image, boxes[i, :], caption)

	drawn_image = PIL.Image.fromarray(image)
	if show_image:
		plt.imshow(drawn_image)
		plt.show()
	return drawn_image


def test_draw_boxes_with_scores():
	from torch.utils.data import DataLoader
	from PIL import Image
	from visdial.data import VisDialDataset
	from configs.attn_disc_lstm_config import get_attn_disc_lstm_config

	config = get_attn_disc_lstm_config()

	dir_path = '/home/quanguet/datasets/visdial/raw_images/VisualDialog_val2018'
	pattern = 'VisualDialog_val2018_000000{:06d}.jpg'

	dataset = VisDialDataset(config, split='test', is_detectron=True)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
	for batch in dataloader:
		break

	img_id = batch['img_ids'][0]
	boxes = batch['boxes'][0].numpy()
	scores = np.ones(len(boxes))
	img_path = img_id_to_path(img_id, dir_path, pattern)
	img = Image.open(img_path)
	print(scores)

	return Image.fromarray(draw_boxes_with_scores(np.array(img), boxes, scores, topk=20))