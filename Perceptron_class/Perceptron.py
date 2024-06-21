import numpy as np

class Perceptron():
    """ Implementación del Perceptrón """

    def __init__(self, eta, max_epocas) -> None:
        """ 
        Constuctor del perceptrón

        Args:
            eta (float): Tasa de apredizaje
            max_epocas (int): Máximo de épocas
        """
        self.eta = eta
        self.max_epocas = max_epocas
        self.w = None
        self.error_epocas = None
        self.w_prom = None

    
    def func_activacion(self, x) -> int:
        """
        Función de activación, en este caso se usa la 
        función signo

        Args:
            x (numeric): Valor de entrada 
        Returns:
            int: 1 si el valor de entrada es positivo, 
            -1 de lo contrario
        """
        return (np.where(x >= 0, 1, -1))

    def entrenar(self, X, d) -> None:
        """
        Función de entrenamiento del perceptrón

        Args:
            X (array[array[float]]): Datos de entrada del entrenamiento
            d (array[int]): Valores esperados de salida
        """
        X = np.c_[np.ones(X.shape[0]), X] 
        self.w = np.random.uniform(-0.5, 0.5, X.shape[1])
        self.error_epocas = [0]*self.max_epocas

        for epoca in range(self.max_epocas):
        
            for i in range(len(X)):

                y = self.func_activacion(np.dot(X[i], self.w))

                if (y != d[i]):
                    self.w += self.eta * (d[i] - y) * X[i]
                    self.error_epocas[epoca] += 1

            if (self.error_epocas[epoca]) == 0:
                self.error_epocas = self.error_epocas[:epoca+1]
                break

    def entrenamiento_promediado(self, X, d) -> None:
        """
        Función de entrenamiento del perceptrón con la variante de promedio

        Args:
            X (array[array[float]]): Datos de entrada del entrenamiento
            d (array[int]): Valores esperados de salida
        """
        X = np.c_[np.ones(X.shape[0]), X] 
        self.w = np.random.uniform(-0.5, 0.5, X.shape[1])
        self.w_prom = np.array([0.0]*X.shape[1])
        self.error_epocas = [0]*self.max_epocas

        contador = 0
        for epoca in range(self.max_epocas):
        
            for i in range(len(X)):

                y = self.func_activacion(np.dot(X[i], self.w))

                if (y != d[i]):
                    self.w += self.eta * (d[i] - y) * X[i]
                    self.error_epocas[epoca] += 1
                
                self.w_prom += self.w
                contador += 1

            if (self.error_epocas[epoca]) == 0:
                self.error_epocas = self.error_epocas[:epoca+1]
                self.w = self.w_prom / contador
                break

    def entrenamiento_MIRA(self, X, d) -> None:
        """
        Función de entrenamiento del perceptrón con la variante MIRA

        Args:
            X (array[array[float]]): Datos de entrada del entrenamiento
            d (array[int]): Valores esperados de salida
        """
        X = np.c_[np.ones(X.shape[0]), X] 
        self.w = np.random.uniform(-0.5, 0.5, X.shape[1])
        self.error_epocas = [0]*self.max_epocas

        for epoca in range(self.max_epocas):
        
            for i in range(len(X)):

                y = self.func_activacion(np.dot(X[i], self.w))

                if (y != d[i]):
                    self.w += ((d[i]- np.dot(X[i], self.w)) / (np.linalg.norm(X[i]) ** 2)) * X[i]
                    self.error_epocas[epoca] += 1

            if (self.error_epocas[epoca]) == 0:
                self.error_epocas = self.error_epocas[:epoca+1]
                break

    def evaluar(self, X) -> int:
        """
        Función que evalúa un conjunto de datos y los clasifica

        Args:
            X (array[array[float]]): Datos a clasificar

        Returns:
            int: Clasificación de los datos dados
        """

        X = np.append([1], X)
        return (self.func_activacion(np.dot(X, self.w)))
     