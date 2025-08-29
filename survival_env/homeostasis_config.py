class HomeostasisConfig:
    def __init__(
            self,
            monster_damage = -0.2,
            food_energy_recovery = 0.5,
            medicine_integrity_recovery = 1,
            basal_metabolic_cost = -0.001,
            action_metabolic_cost = -0.01,
            action_integrity_recovery = 0.01
    ):
        self.monster_damage = monster_damage
        self.food_energy_recovery = food_energy_recovery
        self.medicine_integrity_recovery = medicine_integrity_recovery
        self.basal_metabolic_cost = basal_metabolic_cost
        self.action_metabolic_cost = action_metabolic_cost
        self.action_integrity_recovery = action_integrity_recovery
