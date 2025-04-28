import pygame
import numpy as np
import heapq

# 1. Pedir al usuario las dimensiones
FILAS = int(input("Ingrese número de filas: "))
COLUMNAS = int(input("Ingrese número de columnas: "))

# 2. Ajustar tamaño de ventana
ANCHO_VENTANA = 600
ALTO_VENTANA = 600
ANCHO_NODO = ANCHO_VENTANA // COLUMNAS
ALTO_NODO = ALTO_VENTANA // FILAS

VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA + 50))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

initial_position = {
    "x": None,
    "y": None,
}

final_position = {
    "x": None,
    "y": None,
}

class Nodo:
    def __init__(self, fila, col, ancho_nodo, alto_nodo, total_filas, total_columnas):
        self.fila = fila
        self.col = col
        self.x = col * ancho_nodo  # X se basa en la columna
        self.y = fila * alto_nodo  # Y se basa en la fila
        self.color = BLANCO
        self.ancho = ancho_nodo
        self.alto = alto_nodo
        self.total_filas = total_filas
        self.total_columnas = total_columnas

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA
        
    def do_route(self):
        self.color = VERDE

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.alto))

def precios(matriz, inicio, meta):
    filas, columnas = len(matriz), len(matriz[0])
    
    movimientos = [
        (-1, 0, 10), (-1, 1, 14), (0, 1, 10), 
        (1, 1, 14), (1, 0, 10), (1, -1, 14), 
        (0, -1, 10), (-1, -1, 14)
    ]
    
    def heuristica(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return 15 * min(dx, dy) + 10 * abs(dx - dy)
    
    lista_abierta = [(0, inicio)]
    de_donde_viene = {}
    
    puntaje_g = {pos: float('inf') for fila in range(filas) for pos in [(fila, col) for col in range(columnas)]}
    puntaje_f = {k: v for k, v in puntaje_g.items()}
    
    puntaje_g[inicio] = 0
    puntaje_f[inicio] = heuristica(inicio, meta)
    
    conjunto_cerrado = set()
    
    while lista_abierta:
        current_f, actual = heapq.heappop(lista_abierta)
        
        yield ('current', actual, lista_abierta.copy(), conjunto_cerrado.copy())
        
        if actual == meta:
            camino = []
            while actual in de_donde_viene:
                camino.append(actual)
                actual = de_donde_viene[actual]
            camino.append(inicio)
            camino = camino[::-1]
            yield ('path', camino)
            return
        
        conjunto_cerrado.add(actual)
        
        for dx, dy, costo in movimientos:
            vecino = (actual[0] + dx, actual[1] + dy)
            
            if 0 <= vecino[0] < filas and 0 <= vecino[1] < columnas:
                if matriz[vecino[0]][vecino[1]] in {'C', 'I', 'F'} and vecino not in conjunto_cerrado:
                    puntaje_g_tentativo = puntaje_g[actual] + costo
                    
                    if puntaje_g_tentativo < puntaje_g[vecino]:
                        de_donde_viene[vecino] = actual
                        puntaje_g[vecino] = puntaje_g_tentativo
                        puntaje_f[vecino] = puntaje_g_tentativo + heuristica(vecino, meta)
                        heapq.heappush(lista_abierta, (puntaje_f[vecino], vecino))
        
        yield ('updated', actual, lista_abierta.copy(), conjunto_cerrado.copy())
    
    yield ('no_path', None)

def crear_ventana(filas, columnas, ancho_ventana, alto_ventana):
    ancho_nodo = ancho_ventana // columnas
    alto_nodo = alto_ventana // filas
    return [[Nodo(i, j, ancho_nodo, alto_nodo, filas, columnas) for j in range(columnas)] for i in range(filas)]

def dibujar_ventana(ventana, filas, columnas, ancho_ventana, alto_ventana):
    ancho_nodo = ancho_ventana // columnas
    alto_nodo = alto_ventana // filas
    # Dibujar líneas verticales
    for col in range(columnas + 1):
        x = col * ancho_nodo
        pygame.draw.line(ventana, GRIS, (x, 0), (x, alto_ventana))
    # Dibujar líneas horizontales
    for fila in range(filas + 1):
        y = fila * alto_nodo
        pygame.draw.line(ventana, GRIS, (0, y), (ancho_ventana, y))

def dibujar_boton_inicio(ventana, ancho):
    fuente = pygame.font.SysFont(None, 40)
    texto = fuente.render("comenzar", True, BLANCO)
    boton_rect = pygame.Rect((ancho // 2) - 50, ancho + 10, 100, 40)
    pygame.draw.rect(ventana, AZUL, boton_rect)
    ventana.blit(texto, (boton_rect.x + 25, boton_rect.y + 5))
    return boton_rect

def dibujar(ventana, grid, filas, columnas, ancho_ventana, alto_ventana):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_ventana(ventana, filas, columnas, ancho_ventana, alto_ventana)
    boton_rect = dibujar_boton_inicio(ventana, ancho_ventana)
    pygame.display.update()
    return boton_rect

def obtener_click_pos(pos, filas, columnas, ancho_ventana, alto_ventana):
    ancho_nodo = ancho_ventana // columnas
    alto_nodo = alto_ventana // filas
    x, y = pos
    col = x // ancho_nodo
    fila = y // alto_nodo
    return fila, col

def get_grid_formated(grid):
    color_map = {
        BLANCO: 'C',
        NEGRO: 'P',
        NARANJA: 'I',
        PURPURA: 'F'
    }
    current_grid = [
        [color_map.get(nodo.color, '?') for nodo in fila] 
        for fila in grid
    ]
    return current_grid

def main(ventana, filas, columnas, ancho_ventana, alto_ventana):
    pygame.init()
    pygame.font.init()
    grid = crear_ventana(filas, columnas, ancho_ventana, alto_ventana)
    inicio, fin = None, None
    corriendo = True
    algorithm_generator = None
    path_gen = None
    animating_path = False
    last_step_time = 0
    step_delay = 50

    while corriendo:
        current_time = pygame.time.get_ticks()
        boton_rect = dibujar(ventana, grid, filas, columnas, ancho_ventana, alto_ventana)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[1] > alto_ventana:
                    if boton_rect.collidepoint(pos) and inicio and fin:
                        cuadricula = get_grid_formated(grid)
                        inicio_pos = (initial_position['y'], initial_position['x'])
                        fin_pos = (final_position['y'], final_position['x'])
                        algorithm_generator = precios(cuadricula, inicio_pos, fin_pos)
                else:
                    fila, col = obtener_click_pos(pos, filas, columnas, ancho_ventana, alto_ventana)
                    if fila < 0 or fila >= filas or col < 0 or col >= columnas:
                        continue
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                        initial_position["x"], initial_position["y"] = col, fila
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                        final_position["x"], final_position["y"] = col, fila
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                if pos[1] <= alto_ventana:
                    fila, col = obtener_click_pos(pos, filas, columnas, ancho_ventana, alto_ventana)
                    if fila < 0 or fila >= filas or col < 0 or col >= columnas:
                        continue
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

        if algorithm_generator:
            try:
                event = next(algorithm_generator)
                
                if event[0] == 'path':
                    path = event[1]
                    path_gen = iter(path[1:-1])
                    animating_path = True
                    last_step_time = current_time
                    algorithm_generator = None
                else:
                    _, actual, open_list, closed_list = event
                    for row in grid:
                        for node in row:
                            if node.es_inicio() or node.es_fin() or node.es_pared():
                                continue
                            node.color = BLANCO
                    
                    for (f, pos) in open_list:
                        fila, col = pos
                        node = grid[fila][col]
                        if not node.es_inicio() and not node.es_fin() and not node.es_pared():
                            node.color = AZUL
                    
                    for pos in closed_list:
                        fila, col = pos
                        node = grid[fila][col]
                        if not node.es_inicio() and not node.es_fin() and not node.es_pared():
                            node.color = GRIS
                    
                    fila, col = actual
                    node = grid[fila][col]
                    if not node.es_inicio() and not node.es_fin() and not node.es_pared():
                        node.color = ROJO
                        
            except StopIteration:
                algorithm_generator = None

        if animating_path and current_time - last_step_time > step_delay:
            try:
                pos = next(path_gen)
                fila, col = pos
                node = grid[fila][col]
                node.do_route()
                last_step_time = current_time
            except StopIteration:
                animating_path = False

        pygame.display.update()

    pygame.quit()

main(VENTANA, FILAS, COLUMNAS, ANCHO_VENTANA, ALTO_VENTANA)