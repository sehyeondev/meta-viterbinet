from typing import Tuple
from python_code.detectors.VNET.vnet_detector import VNETDetector
from python_code.ecc.rs_main import decode, encode
from python_code.utils.metrics import calculate_error_rates
from python_code.utils.trellis_utils import calculate_states
from python_code.trainers.trainer import Trainer, STEPS_NUM
import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SER_THRESH = 0.02
SELF_SUPERVISED_ITERATIONS = 500


class VNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'ViterbiNet' + channel_state

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = VNETDetector(n_states=self.n_states,
                                     transmission_lengths=self.transmission_lengths)

    def load_weights(self, snr: float, gamma: float):
        """
        Loads detector's weights defined by the [snr,gamma] from checkpoint, if exists
        """
        if os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'):
            print(f'loading model from snr {snr} and gamma {gamma}')
            checkpoint = torch.load(os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'))
            try:
                self.detector.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise ValueError("Wrong run directory!!!")
        else:
            print(f'No checkpoint for snr {snr} and gamma {gamma} in run "{self.run_name}", starting from scratch')

    def gamma_eval(self, gamma: float) -> np.ndarray:
        """
        Evaluation at a single gamma value.
        :return: ser for batch.
        """
        ser_total = np.zeros(len(self.snr_range['val']))

        for snr_ind, snr in enumerate(self.snr_range['val']):
            self.load_weights(snr, gamma)
            ser_total[snr_ind] = self.single_eval(snr, gamma)

        return ser_total

    def select_batch(self, gt_states: torch.LongTensor, soft_estimation: torch.Tensor) -> Tuple[
        torch.LongTensor, torch.Tensor]:
        """
        Select a batch from the input and gt labels
        :param gt_states: training labels, an int from the range of [0,1,...,n_states-1]
        :param soft_estimation: the soft approximation, distribution over states (per word)
        :return: selected batch from the entire "epoch", contains both labels and the NN soft approximation
        """
        rand_ind = torch.multinomial(torch.arange(gt_states.shape[0]).float(),
                                     self.train_minibatch_size).long().to(device)
        return gt_states[rand_ind], soft_estimation[rand_ind]

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_states(self.memory_length, transmitted_words)
        gt_states_batch, input_batch = self.select_batch(gt_states, soft_estimation.reshape(-1, self.n_states))
        loss = self.criterion(input=input_batch, target=gt_states_batch)
        return loss

    def single_eval(self, snr: float, gamma: float) -> float:
        """
        ViterbiNet single eval - either a normal evaluation or on-the-fly online training
        """
        # eval with training
        if self.self_supervised is True:
            return self.online_training(gamma, snr)
        else:
            # a normal evaluation, return evaluation of parent class
            return super().single_eval(snr, gamma)

    def online_training(self, gamma: float, snr: float) -> float:
        self.deep_learning_setup()
        # draw words of given gamma for all snrs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)
        # self-supervised loop
        total_ser = 0
        for count, (transmitted_word, received_word) in enumerate(zip(transmitted_words, received_words)):
            transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)

            # detect
            detected_word = self.detector(received_word, 'val')

            # decode
            decoded_word = [decode(detected_word) for detected_word in detected_word.cpu().numpy()]
            decoded_word = torch.Tensor(decoded_word).to(device)

            # calculate accuracy
            ser, fer, err_indices = calculate_error_rates(decoded_word, transmitted_word)

            # encode word again
            decoded_word_array = decoded_word.int().cpu().numpy()
            encoded_word = torch.Tensor(encode(decoded_word_array).reshape(1, -1)).to(device)
            errors_num = torch.sum(torch.abs(encoded_word - detected_word)).item()
            print('*' * 20)
            print(f'current: {count, ser, errors_num}')

            if ser <= SER_THRESH:
                # run training loops
                for i in range(SELF_SUPERVISED_ITERATIONS):
                    # calculate soft values
                    soft_estimation = self.detector(received_word, 'train')
                    labels = detected_word if ser > 0 else encoded_word
                    self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)

            total_ser += ser
            if (count + 1) % 10 == 0:
                print(f'Self-supervised: {count + 1}/{transmitted_words.shape[0]}, SER {total_ser / (count + 1)}')
        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        return total_ser


if __name__ == '__main__':
    dec = VNETTrainer()
    dec.train()
    # dec.evaluate()
