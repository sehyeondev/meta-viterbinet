from python_code.detectors.BCJR.bcjr_detector import BCJRDetector
from python_code.trainers.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCJRTrainer(Trainer):
    """
    Trainer for the BCJR model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'BCJR' + channel_state

    def initialize_detector(self):
        """
        Loads the BCJR detector
        """
        self.detector = BCJRDetector(n_states=self.n_states,
                                   memory_length=self.memory_length,
                                   transmission_length=self.transmission_lengths['val'],
                                   val_words=self.val_frames * self.subframes_in_frame,
                                   channel_type=self.channel_type,
                                   noisy_est_var=self.noisy_est_var,
                                   fading=self.fading_in_decoder,
                                   fading_taps_type=self.fading_taps_type,
                                   channel_coefficients=self.channel_coefficients)

    def load_weights(self, snr: float, gamma: float):
        pass

    def train(self):
        raise NotImplementedError("No training implemented for this decoder!!!")


if __name__ == '__main__':
    dec = BCJRTrainer()
    dec.evaluate()
