import argparse


class App:

    def __init__(self):
        self._args = None

    def set_config(self):
        parser = argparse.ArgumentParser(
            description="Ejecutar el agente homeostático con imaginación en el entorno de supervivencia"
        )
        parser.add_argument(
            "--episodes",
            type=int, default=100,
            help="Número de episodios"
        )
        parser.add_argument(
            "--render",
            action="store_true",
            help="Renderizar el entorno durante la ejecución"
        )
        parser.add_argument(
            "--model-path",
            type=str,
            help="Ruta al modelo guardado que se quiere ejecutar"
        )
        parser.add_argument(
            "--save-path",
            type=str,
            default="models",
            help="Ruta para guardar modelos"
        )

        self._args = parser.parse_args()

    def run(self):
        return