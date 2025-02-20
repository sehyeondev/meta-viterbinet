from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
import numpy as np
import torch
import torch.nn as nn
import math

from python_code.utils.trellis_utils import create_transition_table, acs_block

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class BCJRDetector(nn.Module):
    """
    This module implements the classic VA detector
    """

    def __init__(self,
                 n_states: int,
                 memory_length: int,
                 transmission_length: int,
                 val_words: int,
                 channel_type: str,
                 noisy_est_var: float,
                 fading: bool,
                 fading_taps_type: int,
                 channel_coefficients: str):

        super(BCJRDetector, self).__init__()
        self.memory_length = memory_length
        self.transmission_length = transmission_length
        self.val_words = val_words
        self.n_states = n_states
        self.channel_type = channel_type
        self.noisy_est_var = noisy_est_var
        self.fading = fading
        self.fading_taps_type = fading_taps_type
        self.channel_coefficients = channel_coefficients
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device).long()


    def compute_state_priors(self, h: np.ndarray) -> torch.Tensor:
        all_states_decimal = np.arange(self.n_states).astype(np.uint8).reshape(-1, 1)
        all_states_binary = np.unpackbits(all_states_decimal, axis=1).astype(int)
        if self.channel_type == 'ISI_AWGN':
            all_states_symbols = BPSKModulator.modulate(all_states_binary[:, -self.memory_length:])
        else:
            raise Exception('No such channel defined!!!')
        state_priors = np.dot(all_states_symbols, h[:, ::-1].T) # CHECK: convolution is needed
        return torch.Tensor(state_priors).to(device)

    def compute_likelihood_priors(self, y: torch.Tensor, snr: float, gamma: float, phase: str, count: int = None):
        # estimate channel per word (only changes between the h's if fading is True)
        self.h = np.concatenate([estimate_channel(self.memory_length, gamma, noisy_est_var=self.noisy_est_var,
                                             fading=self.fading, index=index, fading_taps_type=self.fading_taps_type,
                                             channel_coefficients=self.channel_coefficients[phase]) for index in
                            range(self.val_words)],
                           axis=0)

        if count is not None:
            self.h = self.h[count].reshape(1, -1)

        # compute priors
        self.state_priors = self.compute_state_priors(self.h)

        if self.channel_type == 'ISI_AWGN':
            self.priors = y.unsqueeze(dim=2) - self.state_priors.T.repeat(
                repeats=[y.shape[0] // self.state_priors.shape[1], 1]).unsqueeze(
                dim=1)
            # to llr representation
            sigma = 10 ** (-snr / 10)
            self.priors = self.priors ** 2 / (2 * sigma ** 2) + math.log(math.sqrt(2 * math.pi) * sigma) # CHECK: plus or minus?
        else:
            raise Exception('No such channel defined!!!')
        return self.priors

    def forward(self, y: torch.Tensor, phase: str, snr: float = None, gamma: float = None,
                count: int = None) -> torch.Tensor:
        """
        The forward pass of the BCJR algorithm
        :param y: input values (batch)
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :returns tensor of detected word, same shape as y
        """
        # compute transition likelihood priors
        priors = self.compute_likelihood_priors(y, snr, gamma, phase, count)

        if phase == 'val':
            decoded_word = torch.zeros(y.shape).to(device)
            
            #### BCJR (sum product) ####
            # compute forward probabilities
            alpha = torch.zeros([y.shape[0], self.transmission_length+1, self.n_states]).to(device)
            alpha[:, 0, 0] = 1  # Initialization: start from state 0
            for i in range(1, self.transmission_length+1):
                for state in range(self.n_states):
                    incoming_states = np.where(self.transition_table_array[:, 0] == state)[0].tolist() + \
                                        np.where(self.transition_table_array[:, 1] == state)[0].tolist()
                    gamma = torch.exp(-priors[:, i - 1, state]).unsqueeze(dim=1)
                    alpha[:, i, state] = torch.sum(alpha[:, i - 1, incoming_states] * gamma, dim=1)
                alpha[:, i, :] /= torch.sum(alpha[:, i, :], dim=1, keepdim=True)  # Normalize


            # compute backward probabilities
            beta = torch.zeros([y.shape[0], self.transmission_length+1, self.n_states]).to(device)
            beta[:, -1, 0] = 1  # Initialization: end state equally likely
            for i in range(self.transmission_length-1, -1, -1):
                for state in range(self.n_states):
                    outgoing_states = self.transition_table_array[state]
                    gamma = torch.exp(-priors[:, i, state]).unsqueeze(dim=1)
                    beta[:, i, state] = torch.sum(beta[:, i + 1, outgoing_states] * gamma, dim=1)
                beta[:, i, :] /= torch.sum(beta[:, i, :], dim=1, keepdim=True)  # Normalize


            # compute MAP v1
            decoded_word = torch.zeros([y.shape[0], self.transmission_length])
            for i in range(self.transmission_length):
                up = torch.zeros(y.shape[0])
                down = torch.zeros(y.shape[0])
                for state in range(self.n_states):
                    transition_up = self.transition_table_array[state, 0]
                    transition_down = self.transition_table_array[state, 1]
                    up += alpha[:, i, state] * torch.exp(-priors[:, i, transition_up]) * beta[:, i, transition_up]
                    down += alpha[:, i, state] * torch.exp(-priors[:, i, transition_down]) * beta[:, i, transition_down]
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
            raise NotImplementedError("No implemented training for this decoder!!!")

