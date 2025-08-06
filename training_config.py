class TrainingConfig:

    def __init__(
            self,
            batch_size,
            epochs,
            actor_optimizer,
            critic_optimizer,
            clipping_eps,
            entropy_coef,
            gamma,
            gae_lambda
    ):
        self._batch_size = batch_size
        self._epochs = epochs
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._clipping_eps = clipping_eps
        self._entropy_coef = entropy_coef
        self._gamma = gamma
        self._gaed_lambda = gae_lambda

    def get_batch_size(self):
        return self._batch_size

    def get_epochs(self):
        return self._epochs

    def get_actor_optimizer(self):
        return self._actor_optimizer

    def get_critic_optimizer(self):
        return self._critic_optimizer

    def get_clipping_eps(self):
        return self._clipping_eps

    def get_entropy_coef(self):
        return self._entropy_coef

    def get_gamma(self):
        return self._gamma

    def get_gaed_lambda(self):
        return self._gaed_lambda
