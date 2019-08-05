"""
A Vocabulary maintains a mapping between words and corresponding unique
integers, holds special integers (tokens) for indicating start and end of
sequence, and offers functionality to map out-of-vocabulary words to the
corresponding token.
"""
import json
import os
from typing import List


class Vocabulary(object):
    """
    A simple Vocabulary class which maintains a mapping between words and
    integer tokens. Can be initialized either by word counts from the VisDial
    v1.0 train dataset, or a pre-saved vocabulary mapping.
    Parameters
    ----------
    word_counts_path: str
        Path to a json file containing counts of each word across captions,
        questions and answers of the VisDial v1.0 train dataset.
    min_count : int, optional (default=0)
        When initializing the vocabulary from word counts, you can specify a
        minimum count, and every token with a count less than this will be
        excluded from vocabulary.
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, word_counts_path: str, min_count: int = 5):
        if not os.path.exists(word_counts_path):
            raise FileNotFoundError(
                f"Word counts do not exist at {word_counts_path}"
            )

        with open(word_counts_path, "r") as word_counts_file:
            word_counts = json.load(word_counts_file)

            # form a list of (word, count) tuples and apply min_count threshold
            word_counts = [
                (word, count)
                for word, count in word_counts.items()
                if count >= min_count
            ]
            # sort in descending order of word counts
            word_counts = sorted(word_counts, key=lambda wc: -wc[1])
            words = [w[0] for w in word_counts]

        self.word2index = {}
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        for index, word in enumerate(words):
            self.word2index[word] = index + 4

        self.index2word = {
            index: word for word, index in self.word2index.items()
        }

    @classmethod
    def from_saved(cls, saved_vocabulary_path: str) -> "Vocabulary":
        """Build the vocabulary from a json file saved by ``save`` method.
        Parameters
        ----------
        saved_vocabulary_path : str
            Path to a json file containing word to integer mappings
            (saved vocabulary).
        """
        with open(saved_vocabulary_path, "r") as saved_vocabulary_file:
            cls.word2index = json.load(saved_vocabulary_file)
        cls.index2word = {
            index: word for word, index in cls.word2index.items()
        }

    def convert_tokens_to_ids(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [
            self.index2word.get(index, self.UNK_TOKEN) for index in indices
        ]

    def save(self, save_vocabulary_path: str) -> None:
        with open(save_vocabulary_path, "w") as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def __len__(self):
        return len(self.index2word)


# in Glove600
word_not_founds =  ['<PAD>', '<S>', '</S>', '<UNK>', 'selfie', '20ish', 'babywearing', 'yess', 'idk', 'tannish', 'nno', 'skiis', 'tanish', 'no-', 'pepperonis', 'yrd', 'no*', '*yes', 'noonish', 'nunchuck', '20s-30s', 'yes-', 'goldish', 'frig', 'ywa', 'tshirt', 'yyes', 'yees', 'hahaha', 'yes3', 'frisbe', 'mid-swing', 'yesw', 'yres', 'brocolli', 'yse', 'eys', 'afternoonish', 'silverish', 'trolly', '10ish', 'sorry**', '20-30s', 'yes*', 'yesd', "'is", 'kiteboard', 'no**', '30s-40s', 'tell-', "'no", 'refridgerator', 'drainer', 'yesa', 'no-but', 'bathmat', 'wakeboarder', 'parasails', 't-ball', "'t", 'tee-shirt', 'tyes', 'sandwhich', 'frizbee', 'opps', '25ish', '60ish', 'it-', 'parasailers', 'no-just', 'surfboarding', 'breakroom', 'skii', 'half-eaten', 'deckered', "'what", 'lived-in', 'any1', 'selfies', '*no', 'wet-suit', "'yes", 'hoody', '7ish', 'parasailer', 'dalmation', 'creamish', 'preforming', 'upclose', 'sorry*', 'surfboarders', 'blck', 'loveseats', 'overcasted', 'coldish', 'mturk', 'ytes', 'buliding', '40s-50s', 'eatable', 'white*', 'passanger', 'mid-jump', 'cruller', 'no-it', 'skort', 'delish', 'so-', 'sorry***', 'blowdryer', 'gnar', 'yellow-ish', 'toliet', 'bedskirt', 'white-fish', 'surfboarder', 'concerte', 'black*', 'white-ish', 'yeds', 'long-hair', 'ohhhhh', 'yws', 'amature', 'yes**', '*i', '*there', 'sortof', 'tell*', 'windsocks', 'cleanish', 'see-thru', 'non-veg', '35ish', 'yes-in', "30's-40", 'yes4', 'sorry-', 'peperoni', '9ish', 'freez', 'urnials', 'mid-stride', 'mini-fridge', '5ish', 'yno', 'right-', 'bi-plane', 'stir-fry', 'iwth', 'wakeboards', 'clearish', 'tallish', 'hotplate', 'uhaul', 'doubledecker', '12ish', 'black-', 'gray-blue', 'tanding', 'diffrent', '-but', 'gray-ish', 'footlongs', 'lunchmeat', 'silve', 'danishes', 'knick-knacks', 'barbwire', "20's-30", 'multi-colors', 'paperclips', 'inground', 'yes-one', 'multicolored-', 'small-ish', 'papasan', 'woops', 'no4', 'prolly', '*is', 'counter-top', 'waverunner', 'yesl', 'floaties', 'vegitables', 'probaby', 'mayby', 'UNK']

# in Glove840B
word_not_founds = ['<PAD>', '</S>', '<UNK>', 'no-', 'no*', '*yes', 'yes-', 'ywa', 'yyes', 'yes3', 'frisbe', 'yesw', 'afternoonish', 'sorry**', 'yes*', 'yesd', "'is", 'no**', 'tell-', "'no", 'yesa', 'no-but', "'t", 'it-', 'no-just', 'deckered', "'what", '*no', "'yes", 'parasailer', 'sorry*', 'surfboarders', 'overcasted', 'white*', 'no-it', 'so-', 'sorry***', 'white-fish', 'surfboarder', 'black*', 'yeds', 'yws', 'yes**', '*i', '*there', 'tell*', 'yes-in', "30's-40", 'yes4', 'sorry-', 'urnials', 'right-', 'black-', '-but', "20's-30", 'yes-one', 'multicolored-', '*is', 'yesl']
