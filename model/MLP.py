import numpy as np

class Funcion_Activacion():

    def __init__(self, tipo):
        self.tipo = tipo

    def funcion(self, x):
        """ Ejecuta la función de activación

        Args:
            x: Dato de entrada 

        Returns:
            Número, el cual es la salida de la función 
        """
        if self.tipo == 'sigmoide':
            return (1 / (1 + np.exp(-x)))
        
        elif self.tipo == 'tanh':
            return (np.tanh(x))
        
        elif self.tipo == 'relu':
            return (np.where(x >= 0, x, 0))
        
        else:
            print("Error, el tipo de función proporcionado no existe.")
        
    def derivada(self, x):
        """ Ejecuta la derivada de la función de activación

        Args:
            x: Dato de entrada

        Returns:
            Número, el cual es la salida de la derivada
        """
        if self.tipo == 'sigmoide':
            r = self.funcion(x)
            return (r * (1 - r))
        
        elif self.tipo == 'tanh':
            return (1 - np.tanh(x)**2)
        
        elif self.tipo == 'relu':
            return (np.where(x > 0, 1, 0))

        else:
            print("Error, el tipo de función proporcionado no existe.")

    def clasificar(self, x):
        """ Clasifica el dato dado

        Args:
            x: Dato a clasificar

        Returns:
            Número resultante de la clasificación (clase)
        """

        if self.tipo == 'sigmoide':
            return (np.where(x > 0.5, 1, -1))
        
        elif self.tipo == 'tanh' or self.tipo == 'relu':
            return (np.where(x > 0, 1, -1))

        else:
            print("Error, el tipo de función proporcionado no existe.")


class MLP():
    """ Implementación del Perceptrón Multicapas """

    def __init__(self, n, eta=1e-3, alpha=0, eps=1e-9, k=1, max_epocas=10000):
        """ 
        Constuctor del perceptrón

        Args:
            n: Número de neuronas en las capas ocultas
            eta: Tasa de apredizaje
            alpha: Factor momentum
            eps: Tolerancia al MSE
            k: Número de capas ocultas
            max_epocas: Máximo de épocas
        """
        self.eta = eta
        self.eps = eps
        self.alpha = alpha
        self.max_epocas = max_epocas

        self.n = n
        self.k = k
        self.funcion_capa_oculta, self.funcion_capa_salida = None, None

        self.w, self.w_capa_oculta, self.w_capa_salida = None, None, None
        self.dw_previo_capa_oculta, self.dw_previo_capa_salida = None, None
        self.bias_oculta, self.bias_salida = None, None
        self.prev_bias_oculta, self.prev_bias_salida = None, None
        self.error_por_epoca, self.errores_val = None, None

    def establecer_f(self, funcion_o, funcion_s):
        self.funcion_capa_oculta = Funcion_Activacion(funcion_o)
        self.funcion_capa_salida = Funcion_Activacion(funcion_s)

    def entrenar(self, X, d):
        """
        Función de entrenamiento del perceptrón

        Args:
            X: Datos de entrada del entrenamiento
            d: Valores esperados de salida
        """
        self.w_capa_oculta = np.random.randn(X.shape[1], self.n)
        self.w_capa_salida = np.random.randn(self.n, self.k) 

        self.dw_previo_capa_oculta = np.zeros(
            np.shape(self.w_capa_oculta))
        self.dw_previo_capa_salida = np.zeros(
            np.shape(self.w_capa_salida))
        
        self.bias_oculta = np.zeros((1, self.n))
        self.bias_salida = np.zeros((1, self.k))

        self.prev_bias_oculta = np.zeros(
            np.shape(self.bias_oculta))
        self.prev_bias_salida = np.zeros(
            np.shape(self.bias_salida))

        self.error_por_epoca, self.errores_val = [0]*self.max_epocas, [0]*self.max_epocas
     
        for epoca in range(self.max_epocas):
            for i in range(X.shape[0]):

                en = X[i:i+1]
                y = d[i:i+1]

                # Propagación hacia delante
                n_oculta = en.dot(self.w_capa_oculta) + self.bias_oculta
                y_oculta = self.funcion_capa_oculta.funcion(n_oculta)
                n_salida = y_oculta.dot(self.w_capa_salida) + self.bias_salida
                y_salida = self.funcion_capa_salida.funcion(n_salida)
                    
                # Propagación hacia atrás
                error = y_salida - y
                d_w_salida = y_oculta.T.dot(error * self.funcion_capa_salida.derivada(n_salida))
                d_bias_salida = np.sum(error * self.funcion_capa_salida.derivada(n_salida), axis=0, keepdims=True)
                error_capa_oculta = error.dot(self.w_capa_salida.T) * self.funcion_capa_oculta.derivada(n_oculta)
                d_w_oculta = en.T.dot(error_capa_oculta)
                d_bias_oculta = np.sum(error_capa_oculta, axis=0, keepdims=True)
                
                self.dw_previo_capa_salida = self.eta * d_w_salida + self.alpha * self.dw_previo_capa_salida
                self.w_capa_salida -= self.dw_previo_capa_salida

                self.dw_previo_capa_oculta = self.eta * d_w_oculta + self.alpha * self.dw_previo_capa_oculta
                self.w_capa_oculta -= self.dw_previo_capa_oculta

                self.prev_bias_salida = self.eta * d_bias_salida + self.alpha * self.prev_bias_salida
                self.bias_salida -= self.prev_bias_salida

                self.prev_bias_oculta = self.eta * d_bias_oculta + self.alpha * self.prev_bias_oculta
                self.bias_oculta -= self.prev_bias_oculta

                self.error_por_epoca[epoca] += np.power(error[0], 2) / 2

            self.error_por_epoca[epoca] = self.error_por_epoca[epoca] / X.shape[0]

            if (epoca > 0 and abs(self.error_por_epoca[epoca] - \
                                 self.error_por_epoca[epoca - 1]) < self.eps):
                self.error_por_epoca = self.error_por_epoca[:epoca + 1]
                break

    def clasificar(self, X):
        """
        Función que evalúa un conjunto de datos y los clasifica

        Args:
            X (array[array[float]]): Datos a clasificar

        Returns:
            int: Clasificación de los datos dados
        """
        n_oculta = X.dot(self.w_capa_oculta) + self.bias_oculta
        y_oculta = self.funcion_capa_oculta.funcion(n_oculta)
        n_salida = y_oculta.dot(self.w_capa_salida) + self.bias_salida
        y_salida = self.funcion_capa_salida.clasificar(self.funcion_capa_salida.funcion(n_salida))
    
        return (np.array([i[0] for i in y_salida]))