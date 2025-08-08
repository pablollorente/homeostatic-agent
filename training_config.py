class TrainingConfig:

    def __init__(
            self,
            replay_buffer_size = 2024,
            batch_size = 64,
            epochs = 10,
            lr=1e-4,
            max_grad_norm = 1.0,
            clipping_eps = 0.2,
            entropy_coef = 0.01,
            gamma = 0.99,
            gae_lambda = 0.95
    ):
        self._replay_buffer_size = replay_buffer_size
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr
        self._max_grad_norm = max_grad_norm
        self._clipping_eps = clipping_eps
        self._entropy_coef = entropy_coef
        self._gamma = gamma
        self._gaed_lambda = gae_lambda

    def get_replay_buffer_size(self):
        return self._replay_buffer_size

    def get_batch_size(self):
        return self._batch_size

    def get_epochs(self):
        return self._epochs

    def get_lr(self):
        return self._lr

    def get_max_grad_norm(self):
        return self._max_grad_norm

    def get_clipping_eps(self):
        return self._clipping_eps

    def get_entropy_coef(self):
        return self._entropy_coef

    def get_gamma(self):
        return self._gamma

    def get_gaed_lambda(self):
        return self._gaed_lambda
