import numpy as np

class Adaline():
    """ Implementación del Adaline (LMS) """

    eta: float
    e: float
    max_epocas: int
    w: np.ndarray[float] | None
    error_epocas: list[float]

    def __init__(self, eta: float, e: float, max_epocas: int) -> None:
        """ 
        Constuctor del Adaline

        Args:
            eta (float): Tasa de apredizaje
            e (float): Tolerancia de error
            max_epocas (int): Máximo de épocas
        """
        self.eta = eta
        self.max_epocas = max_epocas
        self.e = e
        self.w = None
        self.error_epocas = None

    def func_activacion(self, x: float) -> int:
        """
        Función de activación, en este caso se usa la 
        función signo

        Args:
            x (numeric): Valor de entrada 
        Returns:
            int: 1 si el valor de entrada es positivo, 
            -1 de lo contrario
        """
        return (np.where(x > 0, 1, -1))
    
    def entrenar(self, 
                 X: np.ndarray[np.ndarray[float]], 
                 d: np.ndarray[float]) -> None:
        """
        Función de entrenamiento del Adaline

        Args:
            X (array[array[float]]): Datos de entrada del entrenamiento
            X_val (array[array[float]]): Datos de entrada para validar pesos de cada época
            d (array[int]): Valores esperados de salida
            d_val (array[int]): Valores esperados de la salida de validación
        """
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(X.shape[1])
        self.error_epocas = [0.0]*self.max_epocas
        delta_w: np.ndarray[float]

        print("Ejecución Adaline con eta ", self.eta)
        for epoca in range(self.max_epocas):
            E = 0
            delta_w = np.zeros(self.w.shape[0])

            for i in range(X.shape[0]):
                y = np.dot(X[i], self.w)
                delta_w += self.eta * (d[i] - y) * X[i]
                E += (d[i] - y)**2

            print(f"Pesos: {self.w}")
            print(f"Resultado esperado: {d}     Resultado obtenido: {self.evaluar(X)}")
            print("---------------------------------------------------------------------------------------")

            self.w += delta_w
            self.error_epocas[epoca] = E / 2

            if (self.error_epocas[epoca] < self.e):
                self.error_epocas = self.error_epocas[:epoca+1]
                break

    def evaluar(self, X: np.ndarray[np.ndarray[float]]) -> int:
        """
        Función que evalúa un conjunto de datos y los clasifica

        Args:
            X (array[array[float]]): Datos a clasificar

        Returns:
            int: Clasificación de los datos dados
        """
        return self.func_activacion(np.dot(X, self.w))