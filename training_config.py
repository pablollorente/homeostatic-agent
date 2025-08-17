class TrainingConfig:

    def __init__(
            self,
            replay_buffer_size = 2**16,
            min_buffer_size = 2**8,
            n_batches = 4,
            batch_size = 32,
            epochs = 4,
            train_per_steps = 500,
            lr=5e-2,
            max_grad_norm = 1.0,
            clipping_eps = 0.2,
            entropy_coef = 1e-5,
            gamma = 0.99,
            gae_lambda = 0.95
    ):
        self._replay_buffer_size = replay_buffer_size
        self._min_buffer_size = min_buffer_size
        self._n_batches = n_batches
        self._batch_size = batch_size
        self._epochs = epochs
        self._train_per_steps = train_per_steps
        self._lr = lr
        self._max_grad_norm = max_grad_norm
        self._clipping_eps = clipping_eps
        self._entropy_coef = entropy_coef
        self._gamma = gamma
        self._gae_lambda = gae_lambda

    def get_replay_buffer_size(self):
        return self._replay_buffer_size

    def get_min_buffer_size(self):
        return self._min_buffer_size

    def get_n_batches(self):
        return self._n_batches

    def get_batch_size(self):
        return self._batch_size

    def get_epochs(self):
        return self._epochs

    def get_train_per_steps(self):
        return self._train_per_steps

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

    def get_gae_lambda(self):
        return self._gae_lambda
