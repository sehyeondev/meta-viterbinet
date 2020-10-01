import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    previous state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(n_states, 2)
    return transition_table


def acs_block(in_prob: torch.Tensor, llrs: torch.Tensor, transition_table, n_states) -> [torch.Tensor,
                                                                                         torch.LongTensor]:
    """
    Viterbi ACS block
    :param in_prob: last stage probabilities, [batch_size,n_states]
    :param llrs: edge probabilities, [batch_size,1]
    :return: current stage probabilities, [batch_size,n_states]
    """
    trellis = (in_prob + llrs)[transition_table.long()]
    reshaped_trellis = trellis.reshape(-1, n_states, 2)
    return torch.min(reshaped_trellis, 2)


def calculate_states(memory_length: int, transmitted_words: torch.Tensor):
    """
    calculates the state for the transmitted words
    :param memory_length: length of channel memory
    :param transmitted_words: channel transmitted words
    :return: vector of length of transmitted_words with values in the range of 0,1,...,n_states-1
    """
    padded = torch.cat([transmitted_words, torch.zeros([transmitted_words.shape[0], memory_length]).to(device)], dim=1)
    blockwise_words = torch.cat([padded[:, i:-memory_length + i] for i in range(memory_length)], dim=0)
    states_enumerator = (2 ** torch.arange(memory_length)).float().reshape(1, -1).to(device)
    gt_states = torch.mm(states_enumerator, blockwise_words).reshape(-1).long()
    return gt_states
