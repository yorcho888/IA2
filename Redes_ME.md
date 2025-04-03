Jorge Cristian Perez Rodriguez
Vamos a utilizar una red CNN, o en otro caso una Multicapa porque son 3 puntos de las 3 dimensiones.
sus partes:
genera marcas en los puntos de la  ara de una persona: entrada
capa oculta: memoriza o aprende los patrones faciales
salida: da la clasificación
2.-Los patrones a utilizar para este son los puntos, para detectar una emoción, necesitamos definir los puntos de una cara neutral, una vez aprendido esto los puntos aquedaran marcados para el entrenamiento y cuando los puntos de la boca se muevan como una sonrisa detectara felicidad, para tristeza comisura hacia abajo de los labios, cejas hacia abajo y parpados entrecerrados, Enojo: cejas hacia adentro y abajo´, labios apretados o frunciendo el ceño
Sorpresa: Ojos muy abiertos, cejas elevadas y boca abierta
3.- la función de activación depende de la red neuronal: como es multicapa utilizaremos softmax y Relu para el entrenamiento
4.-esto es muy relativo, para cada emoción debe cambiar, depende de sus patrones a utilizar las entradas deberán cambiar, por ejemplo si en felicidad se analiza cejas, boca y mejillas y en enojo solo labios y cejas
5.- pueden ser salida binaria, desde el 0m¿¿ hasta el 1, si detecta 80% felicidad que la muestre con un margen de error de 20%
6.-0.1