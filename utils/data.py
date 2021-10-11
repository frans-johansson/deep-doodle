import numpy as np
import torch
from torch.utils.data import Dataset


class SketchDataset(Dataset):
    """
    Represents a dataset of sketches to be used for training, testing or validating the model

    Members:
        original_data: The data loaded from the .npz file in stroke-3 format.
        model_data: The data required by the model, given in stroke-5 format, and truncated
            to a common length.
    """

    def __init__(self, data, seq_len):
        """
        Initialize a sketch dataset from supplied data. Only keep samples shorter than a given
        maximum length, and pad those samples to that length.

        Args:
            data: A NumPy array of data samples in stroke-3 format.
            max_len: Only utilize samples shorter than this length and pad all samples to this length.
        """
        self.original_data = np.copy(data)
        self.model_data = [
            to_stroke_5(data, seq_len)
            for data in self.original_data
            if len(data) <= seq_len
        ]

    def __len__(self):
        """Returns the length, number of samples, of the loaded data"""
        return len(self.model_data)

    def __getitem__(self, index):
        """Returns the sample at the given index"""
        return self.model_data[index].astype("float32")


def load_quickdraw_data(path, seq_len=200):
    """Creates and returns a train, test and validation dataset from the supplied .npz file"""
    # TODO: Find a way to avoid hardcoding the max_len

    data = np.load(path, encoding="latin1", allow_pickle=True)
    return (
        SketchDataset(data["train"], seq_len=seq_len),
        SketchDataset(data["test"], seq_len=seq_len),
        SketchDataset(data["valid"], seq_len=seq_len),
    )


def separate_stroke_params(strokes):
    """
    Separates a batch of stroke-5 representations into batched stroke offsets and penstates resepctively.
    Also returns the length of each stroke based on the first encountered stop state [0, 0, 0, 0, 1] for
    each batch.

    Args:
        strokes: A tensor of batches of stroke-5 representations of strokes. Should have the shape (batch, seq_len, 5)

    Returns
        S, Ns, p: A tensor of just the stroke offsets (batch, seq_len, 2), a list of stroke lengths for each batch, and
            a tensor of pen states (batch, seq_len, 3) respectively
    """
    S = strokes[..., :2]
    Ns = [
        torch.all(S == torch.tensor([0, 0, 0, 0, 1]), dim=1).nonzero()[0].item()
        for S in strokes
    ]
    p = strokes[..., 2:]

    return S, Ns, p


def create_stroke_mask(Ns, seq_len):
    """
    Generates a binary masking tensor from a list of sequence lenghts

    Args:
        Ns: A list of sequence lengths for each sequence in the batch
        seq_len: The length all sequence have been padded to

    Returns:
        A (batch_size, seq_len) binary masking tensor
    """
    mask = torch.zeros((len(Ns), seq_len))
    for i, length in enumerate(Ns):
        mask[i, :length] = 1
    return mask


def to_stroke_5(stroke, max_len):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # Code from https://github.com/magenta/magenta/blob/main/magenta/models/sketch_rnn/utils.py
    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def to_stroke_3(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    # Code from https://github.com/magenta/magenta/blob/main/magenta/models/sketch_rnn/utils.py
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result
