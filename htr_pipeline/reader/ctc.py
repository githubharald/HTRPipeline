from dataclasses import dataclass, field
from itertools import groupby
from typing import List

import numpy as np


@dataclass
class PrefixTreeNode:
    children: dict = field(default_factory=dict)
    is_word: bool = False


class PrefixTree:
    def __init__(self, words: List[str]):
        self.root = PrefixTreeNode()
        self._add_words(words)

    def _add_word(self, text: str):
        node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = PrefixTreeNode()
            node = node.children[c]
            is_last = (i + 1 == len(text))
            if is_last:
                node.is_word = True

    def _add_words(self, words: List[str]):
        for w in words:
            self._add_word(w)

    def _get_node(self, text: str):
        "get node representing given text"
        node = self.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def is_word(self, text: str) -> bool:
        node = self._get_node(text)
        if node:
            return node.is_word
        return False

    def get_next_chars(self, text: str) -> List[str]:
        chars = []
        node = self._get_node(text)
        if node:
            for k in node.children.keys():
                chars.append(k)
        return chars


@dataclass
class Beam:
    text: str
    prob_blank: float
    prod_non_blank: float

    @property
    def prob_total(self) -> float:
        return self.prob_blank + self.prod_non_blank


def ctc_single_word_beam_search(predictions: np.ndarray,
                                chars: List[str],
                                beam_width: int,
                                prefix_tree: PrefixTree):
    res = []
    for batch_idx in range(predictions.shape[1]):
        num_timesteps = predictions.shape[0]
        prev = [Beam('', 1, 0)]

        # go over all time-steps
        for time_idx in range(num_timesteps):
            curr = []  # list of beams at current time-step

            # go over best beams
            best_beams = sorted(prev, key=lambda x: x.prob_total, reverse=True)[:beam_width]  # get best beams
            for beam in best_beams:
                # calc probability that beam ends with non-blank
                pr_non_blank = 0
                if beam.text != '':
                    # char at time-step t must also occur at t-1
                    label_idx = chars.index(beam.text[-1]) + 1
                    pr_non_blank = beam.prod_non_blank * predictions[time_idx, batch_idx, label_idx]

                # calc probability that beam ends with blank
                pr_blank = beam.prob_total * predictions[time_idx, batch_idx, 0]

                # save result
                curr.append(Beam(beam.text, pr_blank, pr_non_blank))

                # extend current beam with characters according to language model
                # nextChars = beam.getNextChars()
                next_chars = prefix_tree.get_next_chars(beam.text)
                for c in next_chars:
                    # extend current beam with new character
                    label_idx = chars.index(c) + 1
                    if beam.text != '' and beam.text[-1] == c:
                        # same chars must be separated by blank
                        pr_non_blank = predictions[time_idx, batch_idx, label_idx] * beam.prob_blank
                    else:
                        # different chars can be neighbours
                        pr_non_blank = predictions[time_idx, batch_idx, label_idx] * beam.prob_total

                        # save result
                    curr.append(Beam(beam.text + c, 0, pr_non_blank))

            # move current beams to next time-step
            prev = curr

        # return most probable beam
        prev = [beam for beam in prev if prefix_tree.is_word(beam.text)]  # only keep words
        best_beams = sorted(prev, key=lambda x: x.prob_total, reverse=True)
        res.append(best_beams[0].text if best_beams else '')

    return res


def ctc_best_path(predictions: np.ndarray, chars: List[str]) -> List[str]:
    # shape of predictions: WxBxC
    res = []
    for b in range(predictions.shape[1]):
        # get char indices along best path
        best_path = np.argmax(predictions[:, b], axis=1)

        # collapse best path (using itertools.groupby), map to chars, join char list to string
        best_path_decoded = [chars[c - 1] for c, _ in groupby(best_path) if c != 0]

        text = ''.join(best_path_decoded)
        res.append(text)

    return res
