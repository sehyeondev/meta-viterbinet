from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator, OnOffModulator
import numpy as np
import torch
import torch.nn as nn

from python_code.utils.trellis_utils import create_transition_table, acs_block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VADetector(nn.Module):
    """
    This implements the VA decoder by unfolding into a neural network
    """

    def __init__(self,
                 n_states: int,
                 memory_length: int,
                 transmission_length: int,
                 channel_type: str,
                 noisy_est_var: float):

        super(VADetector, self).__init__()
        self.start_state = 0
        self.memory_length = memory_length
        self.transmission_length = transmission_length
        self.n_states = n_states
        self.channel_type = channel_type
        self.noisy_est_var = noisy_est_var
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)

    def compute_state_priors(self, h: np.ndarray):
        all_states = np.unpackbits(np.arange(self.n_states).astype(np.uint8).reshape(-1, 1), axis=1).astype(int)
        if self.channel_type == 'ISI_AWGN':
            # modulation
            all_states_symbols = BPSKModulator.modulate(all_states[:, -self.memory_length:])
        elif self.channel_type == 'Poisson':
            # modulation
            all_states_symbols = OnOffModulator.modulate(all_states[:, -self.memory_length:])
        else:
            raise Exception('No such channel defined!!!')
        state_priors = np.dot(all_states_symbols, h.T)
        return torch.Tensor(state_priors).to(device)

    def forward(self, y: torch.Tensor, phase: str, gamma: float):
        """
        The forward pass of the Viterbi algorithm
        :param y: input values (batch)
        """
        # channel_estimate
        h = estimate_channel(self.memory_length, gamma, noisy_est_var=self.noisy_est_var)

        # initialize input probabilities
        in_prob = torch.zeros(self.n_states).to(device)

        # compute priors
        state_priors = self.compute_state_priors(h)
        priors = torch.abs(y - state_priors).T

        if phase == 'val':
            decoded_word = torch.zeros(y.shape[1]).to(device)
            for i in range(self.transmission_length):
                # get the lsb of the state
                decoded_word[i] = torch.argmin(in_prob) % 2
                # run one Vitebi stage
                out_prob, _ = acs_block(in_prob, priors[i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob.reshape(-1)

            return decoded_word
        else:
            raise NotImplementedError("No implemented training for this decoder!!!")
