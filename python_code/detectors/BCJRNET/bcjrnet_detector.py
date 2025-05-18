from python_code.utils.trellis_utils import create_transition_table, acs_block
from typing import Dict
import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN1_SIZE = 100
HIDDEN2_SIZE = 50


class BCJRNETDetector(nn.Module):
    """
    This implements the VA decoder by an NN on each stage
    """

    def __init__(self,
                 n_states: int,
                 memory_length: int,
                 transmission_lengths: Dict[str, int]):

        super(BCJRNETDetector, self).__init__()
        self.memory_length  = memory_length
        self.transmission_lengths = transmission_lengths
        self.transmission_length = transmission_lengths['val']
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN1_SIZE),
                  nn.Sigmoid(),
                  nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN2_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, y: torch.Tensor, phase: str, snr: float = None, gamma: float = None,
                count: int = None) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param y: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # compute priors
        priors = self.net(y.reshape(-1, 1)).reshape(y.shape[0], y.shape[1], self.n_states)

        if phase == 'val':
            #### BCJR (sum product) ####
            # compute forward probabilities
            alpha = torch.zeros([y.shape[0], self.transmission_length+1, self.n_states]).to(device)
            alpha[:, 0, 0] = 1  # Initialization: start from state 0
            for i in range(1, self.transmission_length+1):
                for state in range(self.n_states):
                    incoming_states = np.where(self.transition_table_array[:, 0] == state)[0].tolist() + \
                                        np.where(self.transition_table_array[:, 1] == state)[0].tolist()
                    ## Exact MAP ##
                    prior = torch.exp(-priors[:, i - 1, state]).unsqueeze(dim=1)
                    alpha[:, i, state] = torch.sum(alpha[:, i - 1, incoming_states] * prior, dim=1)
                    ## Log MAP ##
                    # prior = -priors[:, i - 1, state].unsqueeze(dim=1)
                    # alpha[:, i, state] = torch.logsumexp(torch.log(alpha[:, i - 1, incoming_states]) + prior, dim=1)

                alpha[:, i, :] /= torch.sum(alpha[:, i, :], dim=1, keepdim=True)  # Normalize


            # compute backward probabilities
            beta = torch.zeros([y.shape[0], self.transmission_length+1, self.n_states]).to(device)
            beta[:, -1, 0] = 1  # Initialization: end state equally likely
            for i in range(self.transmission_length-1, -1, -1):
                for state in range(self.n_states):
                    outgoing_states = self.transition_table_array[state]
                    ## Exact MAP ##
                    prior = torch.exp(-priors[:, i, state]).unsqueeze(dim=1)
                    beta[:, i, state] = torch.sum(beta[:, i + 1, outgoing_states] * prior, dim=1)
                    ## Log MAP ##
                    # prior = -priors[:, i, state].unsqueeze(dim=1)
                    # beta[:, i, state] = torch.logsumexp(torch.log(beta[:, i + 1, outgoing_states]) + prior, dim=1)
                beta[:, i, :] /= torch.sum(beta[:, i, :], dim=1, keepdim=True)  # Normalize


            # compute MAP v1
            decoded_word = torch.zeros([y.shape[0], self.transmission_length]).to(device)
            for i in range(self.transmission_length):
                up = torch.zeros(y.shape[0]).to(device)
                down = torch.zeros(y.shape[0]).to(device)
                for state in range(self.n_states):
                    transition_up = self.transition_table_array[state, 0]
                    transition_down = self.transition_table_array[state, 1]
                    ## Exact MAP ##
                    up += alpha[:, i, state] * torch.exp(-priors[:, i, transition_up]) * beta[:, i, transition_up]
                    down += alpha[:, i, state] * torch.exp(-priors[:, i, transition_down]) * beta[:, i, transition_down]
                    ## log MAP ##
                    # up += torch.log(alpha[:, i, state]) - priors[:, i, transition_up] + torch.log(beta[:, i, transition_up])
                    # down += torch.log(alpha[:, i, state]) - priors[:, i, transition_down] + torch.log(beta[:, i, transition_down])
                decoded_word[:, i] = torch.where(up < down, 1, 0)

            # # compute MAP v2
            # batch_indices = torch.arange(y.shape[0])
            # current_state = torch.zeros(y.shape[0], dtype=torch.long, device=device)  # Ensure 1D shape
            # decoded_word = torch.zeros([y.shape[0], self.transmission_length], device=device)
            # for i in range(self.transmission_length):
            #     transition_up = self.transition_table[current_state, 0]
            #     transition_down = self.transition_table[current_state, 1]
            #     up = alpha[batch_indices, i, current_state] * torch.exp(-priors[batch_indices, i, transition_up]) * beta[batch_indices, i, transition_up]
            #     down = alpha[batch_indices, i, current_state] * torch.exp(-priors[batch_indices, i, transition_down]) * beta[batch_indices, i, transition_down]
            #     current_state = torch.where(up > down, transition_up, transition_down)
            #     decoded_word[:, i] = torch.where(up > down, 0, 1)

            #     current_state = torch.where(up > down, transition_up, transition_down)
            #     decoded_word[:, i] = torch.where(up > down, 0, 1)

            prepend_word = torch.zeros([y.shape[0], self.memory_length-1]).to(device)
            decoded_word = torch.cat([prepend_word, decoded_word], dim=1)
            return decoded_word[:,:self.transmission_length]
        else:
            return priors
