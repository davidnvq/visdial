import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import pickle
import textwrap
import numpy as np
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib import patches

class Window(tk.Frame):

	def __init__(self, master, file_path, prefix, root_path):
		tk.Frame.__init__(self, master)
		self.master = master
		self.root_path = root_path
		self.prefix = prefix
		self.file_path = file_path
		self.img_ids = None
		self.img_dialogs = None
		self.total_epochs = None
		self.total_quests = 10
		self.img_index = 0
		self.epoch_index = 0
		self.quest_index = 0
		self.attn_counter = 0

		self.img = None
		self.img_label = None
		self.caption = None
		self.dialog = None
		self.rel_bar = None
		self.rel_ax = None
		self.rank_ax = None
		self.dynamic_axes = None
		self.rel_canvas = None
		self.rank_canvas = None
		self.dynamic_canvas = None
		self.on_files()
		self.init_window()

	def on_files(self, _event=None):
		with open(self.file_path, 'rb') as f:
			self.img_ids, self.img_dialogs = pickle.load(f)
			self.total_epochs = len(self.img_dialogs[self.img_ids[0]]['scores'][0])

	def init_window(self):
		self.frame_img = tk.Frame(self.master, background="#FFFFFF", bd=1, relief="sunken")
		self.frame_rank = tk.Frame(self.master, background="#FFFFFF", bd=1, relief="sunken")
		self.frame_dialog = tk.Frame(self.master, background="#FFFFFF", bd=1, relief="sunken")
		self.frame_dynamic = tk.Frame(self.master, background="#FFFFFF", bd=1, relief="sunken")
		self.frame_relevance = tk.Frame(self.master, background="#FFFFFF", bd=1, relief="sunken")

		self.frame_img.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
		self.frame_rank.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
		self.frame_dialog.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
		self.frame_dynamic.grid(row=0, column=2, sticky="nsew", padx=2, pady=2, rowspan=2)
		self.frame_relevance.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

		self.master.grid_rowconfigure(0, weight=1)
		self.master.grid_rowconfigure(1, weight=2)
		self.master.grid_columnconfigure(0, weight=1)
		self.master.grid_columnconfigure(1, weight=1)
		self.master.grid_columnconfigure(2, weight=1)

		self.master.bind('<Left>',  lambda event: self.on_image(is_next=False))
		self.master.bind('<Right>', lambda event: self.on_image(is_next=True))
		self.master.bind('<Down>',  lambda event: self.on_epoch(is_next=False))
		self.master.bind('<Up>',    lambda event: self.on_epoch(is_next=True))
		self.master.bind('a',       lambda event: self.on_quest(is_next=False))
		self.master.bind('d',       lambda event: self.on_quest(is_next=True))
		self.master.bind('r',       lambda event: self.on_files())

		self.master.bind('1', lambda event: self.to_quest(quest_index=0))
		self.master.bind('2', lambda event: self.to_quest(quest_index=1))
		self.master.bind('3', lambda event: self.to_quest(quest_index=2))
		self.master.bind('4', lambda event: self.to_quest(quest_index=3))
		self.master.bind('5', lambda event: self.to_quest(quest_index=4))
		self.master.bind('6', lambda event: self.to_quest(quest_index=5))
		self.master.bind('7', lambda event: self.to_quest(quest_index=6))
		self.master.bind('8', lambda event: self.to_quest(quest_index=7))
		self.master.bind('9', lambda event: self.to_quest(quest_index=8))

		self.on_image()

	def reset_ques_index(self):
		self.quest_index = self.img_dialogs[self.img_ids[self.img_index]]['round_id'] - 1

	def set_index(self, index, max_value, is_next=True):
		return (index + 1) % max_value if is_next else (index - 1) % max_value

	def resize_img(self, img, max_size=600):
		w, h = img.size
		if w > h:
			return img.resize((max_size, int(max_size * h / w)))
		else:
			return img.resize((int(max_size * w / h), max_size))

	def get_img_path(self, img_name):
		return os.path.join(self.root_path, img_name)

	def get_img_name(self, img_id):
		return self.prefix + str(img_id).zfill(12) + '.jpg'

	def get_img(self, img_index):
		img_id = self.img_ids[img_index]
		img_name = self.get_img_name(img_id)
		img_path = self.get_img_path(img_name)
		img = Image.open(img_path)
		return img

	def get_caption(self, img_index):
		char_limit = 80
		caption = ' '.join(self.img_dialogs[self.img_ids[img_index]]['caption'])
		caption = caption.capitalize()
		wrapper = textwrap.TextWrapper(width=char_limit)
		lines = wrapper.wrap(text=caption)
		caption = '\n'.join(lines)
		return caption


	def make_img_label(self, img, img_index):
		total_imgs = len(self.img_ids)
		render = ImageTk.PhotoImage(img)
		if self.img is not None:
			self.img.destroy()
			self.img_label.destroy()
		self.img = ttk.Label(self.frame_img, image=render, background='#fff')
		self.img.image = render
		self.img.pack(side='top')
		self.img.configure(anchor="center")
		self.img_label = ttk.Label(self.frame_img, text='{}/{}'.format(img_index + 1, total_imgs), background='#fff')
		self.img_label.pack(side='bottom')
		self.img_label.configure(anchor='center')

	def show_img(self, img_index):
		img = self.get_img(img_index)
		img = self.resize_img(img)
		self.make_img_label(img, img_index)

	def show_dialog(self, img_index, ques_index, epoch_index):
		"""
		"""
		char_limit = 80
		dialog = self.img_dialogs[self.img_ids[img_index]]['dialog']
		questions, answers = dialog

		text = '\n\n  Caption:\n  ' + self.get_caption(img_index) + "\n\n"
		text += '  ----' + '-' * char_limit + '\n\n'
		text += '  Dialog\n\n'

		for i in range(ques_index + 1):
			text += '  Q{}: {}\n'.format(i + 1, questions[i][:char_limit].capitalize()) # 80 characters
			text += '  A{}: {}\n'.format(i + 1, answers[i][:char_limit].capitalize())

		text += '  ----' + '-' * char_limit + '\n'

		opt_preds = self.img_dialogs[self.img_ids[img_index]]['scores'][ques_index][epoch_index]

		target_score = opt_preds.get(answers[ques_index])
		max_score = opt_preds[max(opt_preds, key=opt_preds.get)]

		if answers[ques_index] == max(opt_preds, key=opt_preds.get):
			text += '  CORRECT with score = {:.2f}\n\n'.format(max_score)

		elif opt_preds.get(answers[ques_index]) is not None:
			sorted_opts = sorted(opt_preds, key=opt_preds.get, reverse=True)
			for i, opt in enumerate(sorted_opts):
				if opt == answers[ques_index]:
					rank = i + 1
					break

			text += '  In top - [{}] with target score = {:.2f}'.format(rank, target_score)
			text += ' while max score = {:.2f}\n'.format(max_score)
			text += '  Incorrect A: {}\n\n'.format(max(opt_preds, key=opt_preds.get)[:char_limit])
		else:
			text += '  INCORRECT answer'.format(target_score)
			text += ' while max score = {:.2f}\n'.format(max_score)
			text += '  ----' + '-' * char_limit + '\n\n'
			text += '  Target answer: index-{}: {}'.format(ques_index, answers[ques_index][:char_limit].capitalize())
			text += '  Incorrect Answer: {}\n\n'.format(max(opt_preds, key=opt_preds.get)[:char_limit].capitalize())

		if self.dialog is not None:
			self.dialog.destroy()

		self.dialog = ttk.Label(self.frame_dialog, background='#fff', text=text)
		self.dialog.pack(side='top')

	def show_relevance(self, img_index, ques_index, epoch_index):
		fig = Figure(figsize=(5,4))

		if self.rel_ax is not None:
			self.rel_ax.clear()
		else:
			self.rel_ax = fig.add_subplot(111)

		rel_scores = self.img_dialogs[self.img_ids[img_index]]['rel_scores']
		rel_texts = self.img_dialogs[self.img_ids[img_index]]['rel_texts']

		rel_round_id = self.img_dialogs[self.img_ids[img_index]]['round_id']
		rel_pred = self.img_dialogs[self.img_ids[img_index]]['rel_preds'][epoch_index]

		num_opts = np.arange(len(rel_scores))

		labels = [text[:80].capitalize() for text in rel_texts]

		self.rel_ax.bar(num_opts, rel_scores, width=0.2,
		                color='green', align='center', label='GT')
		self.rel_ax.bar(num_opts + 0.2, rel_pred, width=0.2, color='orange',
		                align='center', label='Pred epoch{}'.format(epoch_index))
		self.rel_ax.set_xticks(num_opts)
		self.rel_ax.set_xticklabels(labels, rotation=75, ha='right')

		title = 'Ground Truth & Prediction relevance for question {}, at epoch {}'.format(rel_round_id, epoch_index)
		self.rel_ax.set_title(title)
		self.rel_ax.autoscale(tight=True)
		self.rel_ax.legend()


		fig.tight_layout()

		if self.rel_canvas is None:
			self.rel_canvas = FigureCanvasTkAgg(fig, master=self.frame_relevance)
			self.rel_canvas.get_tk_widget().pack(side='top', fill='x')
		self.rel_canvas.draw()

	def show_rank(self, img_index, ques_index, epoch_index):
		"""
		"""
		self.total_epochs = len(self.img_dialogs[self.img_ids[img_index]]['scores'][ques_index])

		char_limit = 80
		target_answer = self.img_dialogs[self.img_ids[img_index]]['dialog'][1][ques_index]
		opts_dict = self.img_dialogs[self.img_ids[img_index]]['scores'][ques_index][epoch_index]
		fig = Figure(figsize=(5,4))

		if self.rank_ax is not None:
			self.rank_ax.clear()
		else:
			self.rank_ax = fig.add_subplot(111)

		# Draw
		top_opts = sorted(opts_dict, key=opts_dict.get, reverse=True)
		for i, opt in enumerate(top_opts[:10]):
			score = opts_dict[opt]
			color = 'green' if opt == target_answer else 'orange'
			txt_color = 'black'
			# score
			alpha = 0.8 if opt == target_answer else score + 0.1
			self.rank_ax.add_patch(patches.Rectangle((0, 0.1 * i), (score + 0.01), 0.1,
			                                         color=color, alpha=alpha, facecolor='none'))
			# option
			self.rank_ax.annotate('({:.2f}) {}'.format(score, opt[:char_limit].capitalize()),
			                      (0.05, 0.1 * i + 0.05), color=txt_color,
			                      weight=None, fontsize=8, ha='left', va='center')

		self.rank_ax.set_title('Epoch {}/{} top 10 prediction for the question No. {}'.format(
				epoch_index + 1, self.total_epochs, ques_index + 1))
		self.rank_ax.set_yticklabels([])
		fig.tight_layout()

		if self.rank_canvas is None:
			self.rank_canvas = FigureCanvasTkAgg(fig, master=self.frame_rank)
			self.rank_canvas.get_tk_widget().pack(side='top', fill='x')
		self.rank_canvas.draw()


	def show_dynamics(self, img_index, ques_index, epoch_index):
		target_answers_preds = []
		for i in range(self.total_quests):
			target_answer = self.img_dialogs[self.img_ids[img_index]]['dialog'][1][i]
			cur_answer_preds = []
			for epoch in range(self.total_epochs):
				opts_dict = self.img_dialogs[self.img_ids[img_index]]['scores'][i][epoch]
				target_answer_pred = opts_dict[target_answer] if opts_dict.get(target_answer) else 0
				cur_answer_preds.append(target_answer_pred)

			target_answers_preds.append(cur_answer_preds)

		self.dynamic_fig = Figure(figsize=(6, 12))

		if self.dynamic_axes is not None:
			for ax in self.dynamic_axes:
				ax.clear()
		else:
			self.dynamic_axes = self.dynamic_fig.subplots(self.total_quests, 1, sharex=True, sharey=True)

		# Plot each axes
		for i, ax in enumerate(self.dynamic_axes.ravel()):
			if i == ques_index:
				color = 'r'
			else:
				color = 'g'

			ax.plot(list(range(self.total_epochs)),
			        target_answers_preds[i],
			        marker='o', color=color, markersize=2, linewidth=0.5)

			target_answer = self.img_dialogs[self.img_ids[img_index]]['dialog'][1][i]

			ax.set_title('Answer {}: {}'.format(i + 1, target_answer.capitalize()), fontsize=8)
			ax.yaxis.set_ticks_position('none')

			# draw epoch line
			ax.axvline(x=(epoch_index),
			           color='black',
			           linestyle='--',
			           linewidth=1,
			           label='Epoch {}'.format(epoch_index + 1))
			ax.legend()

		self.dynamic_fig.tight_layout()

		if self.dynamic_canvas is None:
			self.dynamic_canvas = FigureCanvasTkAgg(self.dynamic_fig, master=self.frame_dynamic)
			self.dynamic_canvas.get_tk_widget().pack(side='top', fill='both')
		self.dynamic_canvas.draw()

	def show_others(self):
		self.show_rank(self.img_index, self.quest_index, self.epoch_index)
		self.show_dialog(self.img_index, self.quest_index, self.epoch_index)
		self.show_dynamics(self.img_index, self.quest_index, self.epoch_index)
		self.show_relevance(self.img_index, self.quest_index, self.epoch_index)

	def to_quest(self, quest_index):
		if self.quest_index != quest_index:
			self.quest_index = quest_index
			self.show_others()

	def on_image(self, is_next=True):
		self.img_index = self.set_index(self.img_index, len(self.img_ids), is_next)
		self.show_img(self.img_index)
		self.reset_ques_index()
		self.show_others()

	def on_epoch(self, is_next=True):
		self.epoch_index = self.set_index(self.epoch_index, self.total_epochs, is_next)
		self.show_others()

	def on_quest(self, is_next=True):
		self.quest_index = self.set_index(self.quest_index, self.total_quests, is_next)
		self.show_others()

	def client_exit(self):
		exit()


if __name__ == '__main__':
	file_path = 'data/lf_disc_val.pkl'
	prefix = 'VisualDialog_val2018_'
	root_path = '~/datasets/visdial/val/val_images'
	root_path = os.path.expanduser(root_path)
	root = tk.Tk()
	root.geometry("1200x800")
	app = Window(root, file_path, prefix=prefix, root_path=root_path)
	root.mainloop()