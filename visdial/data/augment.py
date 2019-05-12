import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
import random


def augment_word(sentence):
	w2w_dict = [
		{
			'yes' : ['yes', 'yep', 'yea', 'yeah', 'yees'],
			'yep' : ['yes', 'yep', 'yea', 'yeah', 'yees'],
			'yea' : ['yes', 'yep', 'yea', 'yeah', 'yees'],
			'yeah': ['yes', 'yep', 'yea', 'yeah', 'yees'],
			},
		{
			'no'  : ['no', 'nope'],
			'nope': ['no', 'nope'],
			},
		{
			"don't" : ["don't", "do not"],
			'do not': ["don't", "do not"],
			},
		{
			"does not": ["does not", "doesn't"],
			"doesn't" : ["does not", "doesn't"],
			},
		{
			"can't" : ["can't", "can't", "cannot"],
			'cannot': ["can't", "can't", "cannot"],
			},

		{
			"isn't" : ["isn't", "is not"],
			"is not": ["isn't", "is not"],
			},
		{
			"aren't" : ["aren't", "are not"],
			"are not": ["aren't", "are not"],
			},
		{
			"say" : ["say", "tell"],
			"tell": ["say", "tell"],
			},
		{
			"sure"     : ["of course", "for sure", "sure"],
			"of course": ["of course", "for sure", "sure"],
			"for sure" : ["of course", "for sure", "sure"],
			}
		]

	for subdict in w2w_dict:
		if random.random() > 0.5:
			for word in subdict:
				if word in sentence:
					similar_word = subdict[word][random.randint(0, len(subdict[word] - 1))]
					sentence.replace(word, similar_word)
					break
	return sentence


def augment_sentence():
	sent2sent_dict = {
		'is it'   : [
			['yes it is', 'yes', 'yes , it is', 'yea i believe so', 'i think so', 'i believe so', 'yes i think so',
			 'it looks so', 'looks like it is']

			['no', 'no it is not', "i can't tell", "i can't ", "no."]],

		'does she': [
			['nope', 'no', 'not that i can see', "i can't see 1", 'no i do not', "i don't think so",
			 'no , unfortunately']],


		}
	pass
