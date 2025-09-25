# Guía rápida de ejecución
1. Clona el proyecto en tu entorno virtual de Python
2. Instala las dependencias ejecutando en la consola
```sh
    pip install requirements.txt
```
3. Ejcuta en python el script main.py del proyecto con
```sh
    python main.py
```
### Lista de parámetros de ejecución que puedes incluir en la ejecución del main
- `--policy` Indica el tipo de red para la política del agente: feedforward (basic), convolucional (conv), recurrente (recurrent) o convolucional y recurrente (convrec).
- `--intero` Si se indica, el agente utiliza la predicción de la interocepción para generar respuestas condicionadas.
- `--innate` Si se indica, el agente genera respuestas innatas.
- `--imagination` Si se indica, el agente genera imagenes del tablero y estados interoceptivos imaginados que alimentan la red su política.
- `--episodes` Número de episodios hasta finalizar la ejecución.
- `--render` Si se indica, se renderiza el entorno en una ventana para su visualización.
- `--model-path` Ruta donde se encuentra el modelo a ejecutar (no implementado todavía)
- `--save-path` Ruta para guardar el nuevo modelo ejecutado.
- `--buffer-size` Tamaño máximo del buffer de memoria de pasos temporales.
- `--min-buffer-size` Ocupación mínima del buffer para realizar un entrenamiento.
- `--n-batches` Número de mini batches por época de entrenamiento.
- `--batch-size` Tamaño del mini batch de entrenamiento.
- `--epochs` Número de épocas de cada ciclo de entrenamiento.
- `--steps-to-train` Número de steps necesarios que tienen que haberse ejecutado para realizar un entrenamiento desde el último.
- `--lr` Learning rate.
- `--max-grad-norm` Límite superior para realizar el clipping sobre la norma de los gradientes.
- `--clipping-eps` Parámetro para estabelcer el límite inferior y superior para realizar el clipping sobre pérdida del actor en PPO.
- `--entropy-coef` Coefiente para aplicar sobre el bonus de entropía en la pérdida del actor de PPO.
- `--gamma` Factor de descuento de la funcion de utilidad
- `--gae-lambda` Factor de descuento del GAE

## Logs de ejecuciones
El sistema gaurdará en la carpeta `experiments/log` del proyecto un fichero en formato JSON con los parámetros de ejceución y las principales métricas por episodio y ciclo de entrenamiento de cada experimento.
En la carpeta `experiments/plot`, por cada ejecución, en una carpeta que tiene por nombre el identificador de cada experimento, se guardarán las gráficas de las métricas. Estos logs y gráficas se generan al finalizar cada ejecución.