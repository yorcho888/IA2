import pygame
import random
import csv
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model, load_model
from sklearn.model_selection import train_test_split



decision_tree_trained = None
modo_decision_tree = False


neural_network_trained = None
mode_neural_network = False
prediction_counter = 0
directory_to_save_neural_network = r"C:\Users\Jorge\OneDrive\Documentos\semestre9\IAClass\12_projectU2_jump_the_ball_pygames\neural_network"


last_csv_path_saved_for_horizontal_ball = ''
last_csv_path_saved_for_vertical_ball = ''
last_csv_path_saved_for_diagonal_ball = ''

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_manual = False
modo_auto = False  
modo_2_balas = False 
modo_3_balas = False 

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []
datos_modelo_vertical_ball = []
datos_modelo_diagonal_ball = []

# Cargar las imágenes
jugador_frames = [
    pygame.transform.scale(pygame.image.load('C:/Users/Jorge/OneDrive/Documentos/semestre9/IAClass/12_projectU2_jump_the_ball_pygames/N.jpg'), (30, 30)), 

]

bala_img = pygame.image.load('C:/Users/Jorge/OneDrive/Documentos/semestre9/IAClass/12_projectU2_jump_the_ball_pygames/Cara.png')
bala_img = pygame.transform.scale(bala_img, (20, 20))
fondo_img = pygame.image.load('C:/Users/Jorge/OneDrive/Documentos/semestre9/IAClass/12_projectU2_jump_the_ball_pygames/Miley.jpg')
nave_img = pygame.image.load('C:/Users/Jorge/OneDrive/Documentos/semestre9/IAClass/12_projectU2_jump_the_ball_pygames/Nave.jpg')
menu_img = pygame.image.load('C:/Users/Jorge/OneDrive/Documentos/semestre9/IAClass/12_projectU2_jump_the_ball_pygames/menu.jpg')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.image.load('C:/Users/Jorge/OneDrive/Documentos/semestre9/IAClass/12_projectU2_jump_the_ball_pygames/N.jpg')

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# ===== CONSTANTES DE VELOCIDAD =====
VELOCIDAD_BALA_HORIZONTAL = -7  # Velocidad fija hacia izquierda (valor negativo)
VELOCIDAD_BALA_VERTICAL = 5      # Velocidad fija hacia abajo (valor positivo)
INTERVALO_DISPARO = 60           # Frames entre disparos (1 segundo si el juego va a 60FPS)

# Bala horizontal (izquierda)
velocidad_bala = VELOCIDAD_BALA_HORIZONTAL
bala_disparada = False
contador_disparo = 0  # Contador para intervalos

# Bala vertical (arriba->abajo)
bala2 = pygame.Rect(0, 0, 16, 16)
velocidad_bala2 = VELOCIDAD_BALA_VERTICAL
bala2_disparada = False

# Bala diagonal (si usas modo 3 balas)
bala3 = pygame.Rect(w - 16, h//2, 16, 16)  # Posición fija
velocidad_bala3_x = -5  # Velocidad constante diagonal
velocidad_bala3_y = 3
bala3_disparada = False


# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

### ---------------- NEURAL NETWORK ---------------- ###  
### ------------------------------------------------ ###     
def cargar_modelo_neural_network():
    global neural_network_trained
    try:
        model_path = os.path.join(directory_to_save_neural_network, 'neural_network_model.keras')
        neural_network_trained = load_model(model_path)
        print("Modelo de red neuronal cargado exitosamente.")
    except:
        print("No se pudo cargar el modelo de red neuronal")

def predecir_salto_neural_network(velocidad_bala, desplazamiento_bala):
    if neural_network_trained is None:
        print("El modelo de red neuronal no está cargado.")
        return False

    # Preparar los datos de entrada
    input_data = np.array([[velocidad_bala, desplazamiento_bala]])

    # Realizar la predicción
    prediction = neural_network_trained.predict(input_data, verbose=0)
    #prediction = neural_network_trained.predict(input_data)

    # La predicción será un número entre 0 y 1
    # Podemos establecer un umbral, por ejemplo, 0.5
    return prediction[0][0] > 0.5

def generate_neural_network():
    global last_csv_path_saved_for_horizontal_ball, last_csv_path_saved_for_vertical_ball
    global directory_to_save_neural_network

    os.makedirs(directory_to_save_neural_network, exist_ok=True)

    if last_csv_path_saved_for_horizontal_ball == '' or last_csv_path_saved_for_vertical_ball == '':
        print("❌ Faltan uno o ambos datasets.")
        return

    # Cargar datasets
    df1 = pd.read_csv(last_csv_path_saved_for_horizontal_ball)
    df2 = pd.read_csv(last_csv_path_saved_for_vertical_ball)

    if df1.empty or df2.empty:
        print("❌ Alguno de los datasets está vacío.")
        return

    # Convertir el dataset vertical para que tenga una etiqueta estimada
    # Por ejemplo, si la bala está muy cerca, asumimos que hay que saltar
    umbral = 30
    df2['Estatus Salto'] = df2['Desplazamiento Bala Y'].apply(lambda y: 1 if y < umbral else 0)

    # Renombrar columnas para que coincidan
    df2 = df2.rename(columns={'Velocidad Bala': 'Velocidad Bala', 'Desplazamiento Bala Y': 'Desplazamiento Bala'})
    df2 = df2[['Velocidad Bala', 'Desplazamiento Bala', 'Estatus Salto']]
    df1 = df1[['Velocidad Bala', 'Desplazamiento Bala', 'Estatus Salto']]

    # Combinar ambos datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Separar características (X) y etiquetas (y)
    X = df[['Velocidad Bala', 'Desplazamiento Bala']].values
    y = df['Estatus Salto'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo de red neuronal
    model = Sequential([
        Dense(8, input_dim=2, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Precisión del modelo combinado: {accuracy:.2f}")

    # Guardar el modelo
    save_model(model, os.path.join(directory_to_save_neural_network, 'neural_network_model.keras'))
    print("✅ Modelo de red neuronal generado y guardado exitosamente.")

 ### ---------------- DESICITION TREE --------------- ###  
 ### ------------------------------------------------ ###     

### ---------------- DESICITION TREE --------------- ###  
### ------------------------------------------------ ### 
def cargar_modelo_decision_tree():
    global decision_tree_trained
    try:
        decision_tree_trained = joblib.load(directory_to_save_desition_tree + 'decision_tree_model.joblib')
        print("Dasicion tree cargado exitosamente.")
    except:
        print("No se pudo cargar el modelo de árbol de decisión")

def predecir_salto_desition_tree(velocidad_bala, desplazamiento_bala):
    global decision_tree_trained
    if decision_tree_trained is not None:
        prediccion = decision_tree_trained.predict([[velocidad_bala, desplazamiento_bala]])
        print("PREDICción de salto: " + str(prediccion[0]))
        if prediccion[0] == '1':
            print("RETURN DESITION TREE WITH TRUE")
            return True
    return False

def generate_desition_treee():
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    import graphviz
    import joblib

    global last_csv_path_saved_for_horizontal_ball, directory_to_save_desition_tree

    if last_csv_path_saved_for_horizontal_ball == '':
        print('❌ Primero debe de guardar el dataset.')
        return

    # Crear directorio para guardar el árbol de decisión
    directory_to_save_desition_tree = os.path.join(
        os.path.dirname(last_csv_path_saved_for_horizontal_ball),
        'decision_tree_outputs'
    )
    os.makedirs(directory_to_save_desition_tree, exist_ok=True)

    # Leer el CSV sin encabezado
    dataset = pd.read_csv(last_csv_path_saved_for_horizontal_ball, header=None)

    # Validar si hay al menos dos filas (encabezado + datos)
    if len(dataset) < 2:
        print("❌ Dataset inválido: no contiene datos reales.")
        return

    # Eliminar la primera fila (supuesto encabezado incorrecto o basura)
    dataset_cleaned = dataset.iloc[1:].reset_index(drop=True)
    dataset_cleaned = dataset_cleaned.dropna()

    print(f"✅ Filas después de limpieza: {len(dataset_cleaned)}")

    # Validar si hay datos suficientes
    if len(dataset_cleaned) == 0:
        print("❌ Error: No hay suficientes datos para entrenar el modelo.")
        return

    # Guardar el CSV limpio
    cleaned_csv_path = os.path.join(directory_to_save_desition_tree, 'dataset_cleaned.csv')
    dataset_cleaned.to_csv(cleaned_csv_path, index=False, header=False)
    print(f"✅ CSV limpio guardado en: {cleaned_csv_path}")

    # Definir X y y
    try:
        X = dataset_cleaned.iloc[:, :2]  # Características: columnas 0 y 1
        y = dataset_cleaned.iloc[:, 2]   # Etiqueta: columna 2
    except IndexError:
        print("❌ Error: El dataset no tiene suficientes columnas.")
        return

    # Entrenamiento
    try:
        if len(dataset_cleaned) < 5:
            print("⚠️ Pocos datos: se entrenará con todo el dataset.")
            clf = DecisionTreeClassifier()
            clf.fit(X, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ Error al entrenar el modelo: {e}")
        return

    # Exportar y guardar gráfico
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=['V. Bala', 'D. Bala'],
        class_names=['C. 0 (Suelo)', 'C. 1 (Salto)'],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    pdf_path = os.path.join(directory_to_save_desition_tree, 'decision_tree')
    graph.render(pdf_path, format='pdf', cleanup=True)
    print(f"✅ Árbol de decisión guardado como PDF en: {pdf_path}")

    # Mostrar el gráfico (opcional)
    graph.view()

    # Guardar el modelo como joblib
    model_path = os.path.join(directory_to_save_desition_tree, 'decision_tree_model.joblib')
    joblib.dump(clf, model_path)
    print(f"✅ Modelo de árbol de decisión guardado en: {model_path}")

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        bala.x = w - 50  # Posición inicial fija
        bala.y = h - 90
        bala_disparada = True
        velocidad_bala = VELOCIDAD_BALA_HORIZONTAL  # Velocidad constante
# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Posición fija inicial
    bala.y = h - 90
    bala_disparada = False

def disparar_bala2():
    global bala2_disparada, velocidad_bala2
    if not bala2_disparada:
        bala2.x = jugador.x + jugador.width//2 - bala2.width//2  # Centrada sobre el jugador
        bala2.y = 0
        bala2_disparada = True
        velocidad_bala2 = VELOCIDAD_BALA_VERTICAL  # Velocidad constante

# Función para reiniciar la posición de la segunda bala
def reset_bala2():
    global bala2, bala2_disparada
    bala2.x = jugador.x + jugador.width//2 - bala2.width//2  # Mantiene alineación
    bala2.y = 0
    bala2_disparada = False


# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        print("--- manejar_salto ---")
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            print("---manejar_salto > llega a suelo---")
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2
    global modo_decision_tree, salto

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú

    # Mover y dibujar la segunda bala si está en modo 2 o 3 balas
    if modo_2_balas or modo_3_balas:
        if bala2_disparada:
            bala2.y += velocidad_bala2
        else:
            disparar_bala2()

        # Si la bala2 sale de la pantalla, reiniciar su posición
        if bala2.y > h:
            reset_bala2()

        pantalla.blit(bala_img, (bala2.x, bala2.y))

        # Colisión entre la bala2 y el jugador
        if jugador.colliderect(bala2):
            print("Colisión con bala 2 detectada!")
            reiniciar_juego()

    # Mover y dibujar la tercera bala si está en modo 3 balas
    if modo_3_balas:
        if bala3_disparada:
            bala3.x += velocidad_bala3_x
            bala3.y += velocidad_bala3_y
        else:
            disparar_bala3()

        # Si la bala3 sale de la pantalla, reiniciar su posición
        if bala3.x < 0 or bala3.y < 0 or bala3.y > h:
            reset_bala3()

        pantalla.blit(bala_img, (bala3.x, bala3.y))

        # Colisión entre la bala3 y el jugador
        if jugador.colliderect(bala3):
            print("Colisión con bala 3 detectada!")
            reiniciar_juego()

# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, salto, datos_modelo
    global bala, velocidad_bala
    global bala2
    global modo_manual

    if modo_manual:
        pos_x = jugador.x
        dist_horizontal = abs(jugador.x - bala.x)
        dist_vertical = abs(jugador.y - bala2.y)
        salida_salto = 1 if salto else 0
        mover_izq = 1 if pygame.key.get_pressed()[pygame.K_LEFT] else 0
        mover_der = 1 if pygame.key.get_pressed()[pygame.K_RIGHT] else 0

        datos_modelo.append((pos_x, dist_horizontal, dist_vertical, velocidad_bala, salida_salto, mover_izq, mover_der))


# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa, menu_activo
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
        menu_activo = True
        mostrar_menu()
        
        
    else:
        print("Juego reanudado.")

def trace_dataset():
    if(last_csv_path_saved_for_horizontal_ball == ''):
        print("Primero debe de guardar el data set.")  
        print("La ruta inválida es: ", last_csv_path_saved_for_horizontal_ball)
        return  

    df = pd.read_csv(last_csv_path_saved_for_horizontal_ball)

    # Verificar si las columnas no contienen texto. Si es así, convertirlas a numéricas o reemplazarlas con NaN
    df['Velocidad Bala'] = pd.to_numeric(df['Velocidad Bala'], errors='coerce')
    df['Desplazamiento Bala'] = pd.to_numeric(df['Desplazamiento Bala'], errors='coerce')
    df['Estatus Salto'] = pd.to_numeric(df['Estatus Salto'], errors='coerce')

    # Reemplazar valores negativos con su valor absoluto. Antes de esto, eliminamos valores NaN
    df = df.dropna()
    df['Velocidad Bala'] = df['Velocidad Bala'].abs()
    df['Desplazamiento Bala'] = df['Desplazamiento Bala'].abs()

    # Estadísticas de los datos
    print("\n------------ DATA EXAMPLE ------------")
    print(df.head())
    print("\n------------ TYPES OF PARAMS ------------")
    print(df.dtypes)
    print("\n------------ STATISTICS ------------")
    print(df.describe())
    print("\n------------ CORRELATIONS ------------")
    print(df.corr())

    # Crear gráfico 3D y una cuadrícula 3D. Luego, graficar los datos, etiquetas adicionales y finalmente una barra de color
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Desplazamiento Bala'], 
                        df['Velocidad Bala'], 
                        df['Estatus Salto'],
                        c=df['Estatus Salto'], 
                        cmap='viridis')

    ax.set_xlabel('Desplazamiento Bala')
    ax.set_ylabel('Velocidad Bala')
    ax.set_zlabel('Estatus del Salto')
    ax.set_title('Análisis 3D de Datos del Juego')

    plt.colorbar(scatter, label='Estatus Salto')
    plt.show()

    # Gráfico de dispersión 2D.
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Desplazamiento Bala'], df['Velocidad Bala'], c=df['Estatus Salto'], cmap='viridis')
    plt.xlabel('Desplazamiento Bala')
    plt.ylabel('Velocidad Bala')
    plt.title('Desplazamiento vs Velocidad de la Bala')
    plt.colorbar(label='Estatus Salto')
    plt.show()

def save_data_set():
    global last_csv_path_saved_for_horizontal_ball
    global datos_modelo

    directory_to_save_datasets = r"C:\Users\Jorge\OneDrive\Documentos\semestre9\IAClass\12_projectU2_jump_the_ball_pygames\datasets"
    os.makedirs(directory_to_save_datasets, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_multisalida_{timestamp}.csv"
    file_path = os.path.join(directory_to_save_datasets, filename)

    if len(datos_modelo) == 0:
        print("❌ No se puede guardar: no hay datos registrados.")
        return

    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "PosX Jugador",
                "Dist Bala Horizontal",
                "Dist Bala Vertical",
                "Velocidad Bala Horizontal",
                "Saltar",
                "Mover Izquierda",
                "Mover Derecha"
            ])
            for dato in datos_modelo:
                writer.writerow(dato)
        last_csv_path_saved_for_horizontal_ball = file_path
        print(f"✅ Dataset multisalida guardado exitosamente en: {file_path}")
    except Exception as e:
        print(f"❌ Error al guardar el dataset: {e}")


def print_menu_options():
    lineas = [
        "'D' para Auto con Desition Tree",
        "'N' para Auto con Neural Network",
        "'M' para Manual",
        "'F' para entrenar modelos",
        "'G' Para almacenar dataset",
        "'T' Para graficar dataset",
        "Presiona '2' para modo 2 balas",
        "",
        "'Q' para Salir"
    ]
    
    # Posición inicial
    x = w // 4
    y = h // 2 - (len(lineas) * 20)  # Ajusta el desplazamiento vertical según el número de líneas
    
    for linea in lineas:
        texto = fuente.render(linea, True, BLANCO)
        pantalla.blit(texto, (x, y))
        y += 40  
    pygame.display.flip() 
    
def generate_desition_tree_2_balas():
    global last_csv_path_saved_for_horizontal_ball, last_csv_path_saved_for_vertical_ball
    if last_csv_path_saved_for_horizontal_ball == '' or last_csv_path_saved_for_vertical_ball == '':
        print("❌ Faltan datasets horizontal o vertical.")
        return

    # Cargar ambos CSVs
    df1 = pd.read_csv(last_csv_path_saved_for_horizontal_ball)
    df2 = pd.read_csv(last_csv_path_saved_for_vertical_ball)

    # Validar que tengan datos
    if df1.empty or df2.empty:
        print("❌ Uno de los datasets está vacío.")
        return

    # Convertir columna vertical en target binario: 1 si distanciaY < umbral, 0 si no
    umbral = 30
    df2['Estatus Salto'] = df2['Desplazamiento Bala Y'].apply(lambda d: 1 if d < umbral else 0)

    # Unificar datasets horizontal y vertical
    df1 = df1[['Velocidad Bala', 'Desplazamiento Bala', 'Estatus Salto']]
    df2 = df2.rename(columns={'Velocidad Bala': 'Velocidad Bala', 'Desplazamiento Bala Y': 'Desplazamiento Bala'})
    df2 = df2[['Velocidad Bala', 'Desplazamiento Bala', 'Estatus Salto']]
    df = pd.concat([df1, df2], ignore_index=True)

    # Entrenamiento
    X = df[['Velocidad Bala', 'Desplazamiento Bala']]
    y = df['Estatus Salto']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Exportar árbol
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=['V. Bala', 'D. Bala'],
                               class_names=['C. 0', 'C. 1'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    output_dir = os.path.join(os.path.dirname(last_csv_path_saved_for_horizontal_ball), 'decision_tree_outputs')
    os.makedirs(output_dir, exist_ok=True)
    graph.render(os.path.join(output_dir, 'decision_tree_2_balas'), format='pdf', cleanup=True)
    joblib.dump(clf, os.path.join(output_dir, 'decision_tree_2_balas_model.joblib'))

    print("✅ Árbol de decisión entrenado con datos de 2 balas.")

def train_models():
    generate_neural_network()
# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global pausa, menu_activo, modo_auto, modo_manual, modo_2_balas, modo_3_balas
    global modo_decision_tree, modo_manual, modo_auto, mode_neural_network

    pantalla.fill(NEGRO)
    
    print_menu_options()
    
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_d:
                    print("Press d")
                    modo_auto = True
                    modo_decision_tree = True
                    mode_neural_network = False
                    modo_manual = False
                    modo_2_balas = True
                    modo_3_balas = False
                    menu_activo = False
                    pausa = False
                    cargar_modelo_decision_tree()
                    print('- - - - Option auto: desition tree selected - - - -')
                elif evento.key == pygame.K_n:
                    print("Press n")
                    
                    modo_auto = True
                    modo_decision_tree = False
                    mode_neural_network = True
                    modo_manual = False
                    modo_2_balas = True
                    modo_3_balas = False
                    menu_activo = False
                    pausa = False
                    cargar_modelo_neural_network()
                    print('- - - - Option auto: neural network selected - - - -')
                elif evento.key == pygame.K_m:
                    print("Press m")
                    
                    modo_auto = False
                    modo_manual = True
                    modo_auto = False
                    modo_decision_tree = False
                    modo_2_balas = True
                    modo_3_balas = False
                    print("Press m2")
                    menu_activo = False
                    print("Press m3")
                    correr = True
                    pausa = False
                    #main()
                elif evento.key == pygame.K_f:
                    print("Press f")
                    
                    train_models()
                    menu_activo = True
                elif evento.key == pygame.K_g:
                    print("Press g")

                    save_data_set()
                    menu_activo = True
                elif evento.key == pygame.K_t:
                    print("Press t")

                    trace_dataset()
                elif evento.key == pygame.K_2:
                    modo_auto = False
                    modo_manual = False
                    modo_2_balas = True
                    modo_3_balas = False
                    menu_activo = False
                elif evento.key == pygame.K_3:
                    modo_auto = False
                    modo_manual = False
                    modo_2_balas = False
                    modo_3_balas = True
                    menu_activo = False
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

    pygame.display.flip()
# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo, bala2_disparada, bala3_disparada, salto_altura, datos_modelo
    
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    salto_altura = 15  # Restablecer la velocidad de salto
    en_suelo = True
    # Reiniciar la segunda bala
    bala2.x = random.randint(0, w - 16)
    bala2.y = 0
    bala2_disparada = False
    # Reiniciar la tercera bala
    bala3.x = w - 16
    bala3.y = random.randint(0, h - 16)
    bala3_disparada = False
    
    
    # Mostrar los datos recopilados hasta el momento
    print("Datos recopilados para el modelo: ", datos_modelo)
    
    
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo


def run_any_mode(correr):
    global salto, en_suelo, bala_disparada
    global modo_decision_tree, modo_manual, modo_auto
    global bala, velocidad_bala, jugador, prediction_counter
    pygame.display.flip()
    reloj = pygame.time.Clock()
    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    print('saltando.....')
                    salto = True
                    en_suelo = False
                    salto_altura = 15  # Restablecer la velocidad de salto al iniciar un nuevo salto
                
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    print("Juego terminado.")
                    pygame.quit()
                    exit()
            

        if not pausa:
            # Modo manual: el jugador controla el salto
            if not modo_auto:
                print('modo manual')
                if salto:
                    manejar_salto()
                # Guardar los datos si estamos en modo manual
                guardar_datos()
                
            elif modo_decision_tree:
                if decision_tree_trained is not None:
                    desplazamiento_bala1 = bala.x - jugador.x
                    desplazamiento_bala2 = jugador.y - bala2.y

                    pred1 = predecir_salto_desition_tree(velocidad_bala, desplazamiento_bala1)
                    pred2 = predecir_salto_desition_tree(velocidad_bala2, desplazamiento_bala2)

                    if (pred1 or pred2) and en_suelo:
                        print('Saltando... decisión activada por alguna bala.')
                        salto = True
                        en_suelo = False
                if salto:
                    manejar_salto()
                    
            elif mode_neural_network:
                prediction_counter += 1
                if prediction_counter % 1 == 0:
                    if neural_network_trained is not None:
                        desplazamiento_bala1 = bala.x - jugador.x
                        desplazamiento_bala2 = jugador.y - bala2.y

                        pred1 = predecir_salto_neural_network(velocidad_bala, desplazamiento_bala1)
                        pred2 = predecir_salto_neural_network(velocidad_bala2, desplazamiento_bala2)

                        if (pred1 or pred2) and en_suelo:
                            print('Saltando... predicción activada por alguna bala.')
                            salto = True
                            en_suelo = False
                if salto:
                    manejar_salto()

            
            # Move right or left
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                jugador.x -= 5  
            if keys[pygame.K_RIGHT]:
                jugador.x += 5  

            # Mantener al jugador dentro de los límites de la pantalla
            if jugador.x < 0:
                jugador.x = 0
            if jugador.x > w - jugador.width:
                jugador.x = w - jugador.width

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(60)  # Limitar el juego a 60 FPS


def main():
    global salto, en_suelo, bala_disparada
    global modo_decision_tree, modo_manual, modo_auto
    global bala, velocidad_bala, jugador, prediction_counter

    
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True
    run_any_mode(correr)

    pygame.quit()

if __name__ == "__main__":
    main()