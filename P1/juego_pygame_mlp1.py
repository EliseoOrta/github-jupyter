import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Opcional: para graficar los datos en 2D y 3D
import matplotlib
# Configuramos backend para ventanas interactivas (TkAgg funciona en la mayoría de sistemas)
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  # Usa el backend por defecto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, necesario para activar 3D en matplotlib

# Activamos modo interactivo para que las ventanas no bloqueen el juego
plt.ion()


# Ventana base y factor de escala
BASE_W, BASE_H = 640, 480
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1


"""@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    salto: int  # 1 si saltó EN ESE FRAME, 0 si no"""

@dataclass
class Sample:
    velocidad_bala: float
    distancia: float 
    altura_bala: float  # NUEVA VARIABLE
    accion: int         # 0: Quieto, 1: Saltar, 2: Agacharse

class Juego:
    def __init__(self) -> None:
        pygame.init()

        # Ventana fija (sin redimensionamiento automático) para evitar
        # problemas en pantallas muy grandes / 2K / 4K.
        self._flags = 0
        self._fullscreen = False

        # Tamaño fijo de ventana
        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + MLP (solo memoria)")

        # Colores
        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        # Estado global
        self.corriendo = True
        self.modo_auto = False

        # Datos / modelo
        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        # Caso especial: cuando solo hay una clase en los datos
        # (0 = nunca salto, 1 = siempre salto).
        self.clase_unica: Optional[int] = None
        # Debug / info del modelo en tiempo real
        self.ultima_proba_salto: Optional[float] = None

        # Parámetros de decisión
        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        # Geometría / física (se rellenan en _apply_resolution)
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        # Velocidad de desplazamiento del fondo
        self.fondo_speed = 3

        self.salto = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        # Velocidad base de la bala (en píxeles/frame, negativa porque va de der→izq)
        self.velocidad_bala = -12
        self.bala_disparada = False
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- resolución / assets -----------------
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin = int(50 * self.scale)
        ground_offset = int(100 * self.scale)
        self.ground_y = self.h - ground_offset

        self.player_size = (int(64 * self.scale), int(144 * self.scale))
        self.bullet_size = (int(24 * self.scale), int(24 * self.scale))
        self.ship_size = (int(128 * self.scale), int(128 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            #self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size[0], self.player_size[1])
            self.jugador = pygame.Rect(self.margin, 0, self.player_size[0], self.player_size[1])
            self.jugador.bottom = self.ground_y
            """self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0],
                self.bullet_size[1],
            )"""
            self.bala = pygame.Rect(self.w - int(100 * self.scale), 0, self.bullet_size[0], self.bullet_size[1])
            self.bala.bottom = self.ground_y
            """self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0],
                self.ship_size[1],
            )"""
            self.nave = pygame.Rect(self.w - int(100 * self.scale), 0, self.ship_size[0], self.ship_size[1])
            self.nave.bottom = self.ground_y

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/mono_frame_1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_4.png"), self.player_size),
        ]
        self.bala_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size,
            (160, 120, 255, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h),
            (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size,
            (140, 255, 200, 255),
        )

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            # Volver a ventana fija BASE_W x BASE_H
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- estado juego / modelo -----------------
    def _reset_estado_juego(self) -> None:
        self.jugador.height = self.player_size[1]
        #self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.x = self.margin
        self.jugador.bottom = self.ground_y
        #self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.nave.x = self.w - int(100 * self.scale)
        self.nave.bottom = self.ground_y
        #self.bala.x = self.w - self.margin
        self.bala.x = self.w - int(100 * self.scale)
        self.bala.bottom = self.ground_y
        #self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada = False
        self.velocidad_bala = int(-10 * self.scale)
        self.salto = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w
        self.agachado = False
        self.bolsa_carriles = []
        

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    # ----------------- export / gráficas -----------------

    def exportar_datos_csv(self) -> str:
        """
        Exporta el contenido de self.datos_modelo a un CSV sencillo.
        Devuelve un mensaje con la ruta del archivo o el motivo del fallo.
        """
        if not self.datos_modelo:
            return "No hay datos para exportar."

        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")

        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                """writer.writerow(["velocidad_bala", "distancia", "salto"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.salto])"""
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"

        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        """
        Grafica velocidad_bala vs distancia en 2D,
        coloreando por salto (0 / 1).
        Abre una ventana interactiva (desde el hilo principal, no bloqueante).
        """
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        cs = ["red" if s.salto == 1 else "blue" for s in self.datos_modelo]

        # Cerrar figura anterior si existe para evitar acumulación
        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos entrenamiento MLP (rojo=salto, azul=no salto)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        # Mostrar sin bloquear (modo interactivo ya está activado con plt.ion())
        plt.show(block=False)
        plt.draw()  # Forzar actualización de la ventana

        return "Mostrando gráfica 2D interactiva (puedes rotar/zoom)."

    def graficar_datos_3d(self) -> str:
        """
        Grafica velocidad_bala vs distancia vs índice de tiempo (frame) en 3D,
        coloreando por salto (0 / 1).
        Abre una ventana interactiva (desde el hilo principal, no bloqueante).
        """
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = list(range(len(self.datos_modelo)))  # eje "tiempo" aproximado
        cs = ["red" if s.salto == 1 else "blue" for s in self.datos_modelo]

        # Cerrar figura anterior si existe para evitar acumulación
        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()

        # Crear eje 3D correctamente desde la figura
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Índice (tiempo aproximado)")
        ax.set_title("Datos entrenamiento MLP 3D (rojo=salto, azul=no salto)")
        plt.tight_layout()
        # Mostrar sin bloquear (modo interactivo ya está activado con plt.ion())
        plt.show(block=False)
        plt.draw()  # Forzar actualización de la ventana

        return "Mostrando gráfica 3D interactiva (puedes rotar/zoom)."

    # ----------------- bala / salto -----------------
    """def disparar_bala(self) -> None:
        if not self.bala_disparada:
            # Aumentamos ligeramente el rango de velocidad para que el juego sea más rápido.
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.bala_disparada = True"""
    
    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            # Velocidad aleatoria
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            
            # --- LÓGICA DE BOLSA ALEATORIA (TETRIS BAG) ---
            # Si la bolsa está vacía, la llenamos con los 4 carriles y la mezclamos
            if not self.bolsa_carriles:
                self.bolsa_carriles = [0, 1, 2, 3]
                random.shuffle(self.bolsa_carriles)

            # --- LÓGICA DE 4 CARRILES ---
            carril = self.bolsa_carriles.pop()
            #carril = 1
            altura_jugador = self.player_size[1]
            
            if carril == 0:
                # Carril 0: Suelo (Toca los pies) -> Acción: Saltar
                self.bala.bottom = self.ground_y
                
            elif carril == 1:
                # Carril 1: Medio (Cintura) -> Acción: Saltar O Agacharse
                # Lo ponemos un poco más arriba de la mitad del jugador para que 
                # al agacharse (que reduce la hitbox a la mitad), la bala pase por encima.
                offset = (altura_jugador // 2) + int(5 * self.scale)
                self.bala.bottom = self.ground_y - offset
                
            elif carril == 2:
                # Carril 2: Alto (Cabeza) -> Acción: Agacharse
                # Pasa golpeando la parte superior del jugador de pie
                offset = altura_jugador - int(10 * self.scale)
                self.bala.bottom = self.ground_y - offset
                
            elif carril == 3:
                # Carril 3: Cielo -> Acción: Quedarse quieto
                # Pasa totalmente por encima del jugador
                offset = altura_jugador + int(20 * self.scale)
                self.bala.bottom = self.ground_y - offset

            # Detalle visual: Alinear el centro del UFO con la bala para que el disparo tenga sentido
            self.nave.centery = self.bala.centery
            # Asegurarse de que el UFO no se entierre por error si baja mucho
            if self.nave.bottom > self.ground_y:
                self.nave.bottom = self.ground_y

            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo:
            self.salto = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            """if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True"""
            if self.jugador.bottom >= self.ground_y:
                self.jugador.bottom = self.ground_y  # Alineamos los pies al ras del suelo
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True


    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)
        
        # Etiquetamos la acción actual
        if not self.en_suelo:
            accion_label = 1  # 1 = Saltar
        elif self.agachado:
            accion_label = 2  # 2 = Agacharse
        else:
            accion_label = 0  # 0 = Quieto

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                altura_bala=float(self.bala.y),
                accion=accion_label,
            )
        )
    
    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."
        
        # Ahora X tiene 3 variables y usamos s.accion
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        
        """clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            tipo = f"CLASE ÚNICA ({self.clase_unica})"
            return True, f"Modelo trivial entrenado: {tipo}. Faltan otras acciones."""
        
        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            
            # Definimos el nombre de la acción para el mensaje
            nombres = {0: "QUEDARSE QUIETO", 1: "SALTAR", 2: "AGACHARSE"}
            accion_aprendida = nombres.get(self.clase_unica, "DESCONOCIDA")
            
            return True, f"Modelo entrenado: El estilo de juego actual es {accion_aprendida}."
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = MLPClassifier(
            hidden_layer_sizes=(10, 10),
            activation="relu",
            solver="adam",
            max_iter=300000,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True
        return True, f"MLP entrenado. Accuracy test ≈ {acc:.3f}"
    
    def decision_auto(self) -> int:
        if not self.modelo_entrenado or not self.bala_disparada:
            return 0
        distancia = abs(self.jugador.x - self.bala.x)

        if self.clase_unica is not None and self.modelo is None:
            return self.clase_unica

        if self.modelo is None or self.scaler is None:
            return 0

        # Pasamos las 3 variables para predecir
        X = [[float(self.velocidad_bala), float(distancia), float(self.bala.y)]]
        Xs = self.scaler.transform(X)
        
        # Obtenemos la clase predicha (0, 1 o 2)
        pred = int(self.modelo.predict(Xs)[0])
        return pred

    # ----------------- menú -----------------
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO salta)",
            "T - Entrenar MLP",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
        ]
        x0 = int(80 * self.scale)
        y = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad = max(6, int(6 * self.scale))
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'sí' if self.modelo_entrenado else 'no'}",
            f"Resolución: {self.w}x{self.h} | scale≈{self.scale:.2f} | ventana_decisión≈{self.decision_window}",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                # Ya no reaccionamos a cambios de tamaño de ventana,
                # la ventana es fija.
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero entrena el MLP (T) en esta sesión."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ----------------- render / loop -----------------
    def _update_frame(self) -> None:
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0

        #self.pantalla.blit(self.jugador_frames[self.current_frame], (self.jugador.x, self.jugador.y))
        sprite_actual = self.jugador_frames[self.current_frame]
        
        # Si está agachado, aplastamos la imagen dinámicamente
        if self.agachado:
            sprite_actual = pygame.transform.scale(
                sprite_actual, 
                (self.jugador.width, self.jugador.height)
            )

        self.pantalla.blit(sprite_actual, (self.jugador.x, self.jugador.y))
        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        # Si hay colisión, solo reiniciamos el estado del juego
        # pero NO volvemos al menú para evitar el efecto de
        # "se cierra y se abre" constantemente.
        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        # Info del modelo en tiempo real (solo si hay modelo entrenado)
        if self.modelo_entrenado and self.modo_auto and self.ultima_proba_salto is not None:
            txt = self.fuente_chica.render(
                f"proba_salto≈{self.ultima_proba_salto:.2f}", True, self.AMARILLO
            )
            # Esquina superior izquierda, con un pequeño margen.
            self.pantalla.blit(txt, (10, 10))

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            salto_frame = False

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                # La ventana es de tamaño fijo: ignoramos eventos VIDEORESIZE.
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        # Reiniciamos el estado del juego (incluida la bala)
                        # y volvemos al menú.
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif e.key == pygame.K_SPACE and (not self.modo_auto) and self.en_suelo and not self.agachado:
                        salto_frame = True
                        self.iniciar_salto()

            if not self.corriendo:
                break
            
            # --- NUEVA LÓGICA DE AGACHARSE ---
            teclas = pygame.key.get_pressed()
            # Solo se puede agachar si está en el suelo y NO está en modo automático (por ahora)
            if teclas[pygame.K_DOWN] and self.en_suelo and not self.modo_auto:
                if not self.agachado:
                    self.agachado = True
                    # Reducimos la hitbox a la mitad
                    self.jugador.height = self.player_size[1] // 2
                    self.jugador.bottom = self.ground_y
            else:
                if self.agachado:
                    self.agachado = False
                    # Restauramos la hitbox original
                    self.jugador.height = self.player_size[1]
                    self.jugador.bottom = self.ground_y
            # ---------------------------------

            """if self.modo_auto:
                if self.decision_auto_saltar():
                    self.iniciar_salto()
            else:
                # En modo manual registramos SIEMPRE la decisión de este frame.
                # Ahora la etiqueta salto=1 cubre TODO el tiempo en el aire.
                self.registrar_decision_manual()"""
            
            if self.modo_auto:
                accion_predicha = self.decision_auto()
                
                if accion_predicha == 1: # SALTAR
                    if self.agachado: # Si estaba agachado, se levanta para poder saltar
                        self.agachado = False
                        self.jugador.height = self.player_size[1]
                        self.jugador.bottom = self.ground_y
                    self.iniciar_salto()
                    
                elif accion_predicha == 2: # AGACHARSE
                    if self.en_suelo and not self.agachado:
                        self.agachado = True
                        self.jugador.height = self.player_size[1] // 2
                        self.jugador.bottom = self.ground_y
                        
                else: # 0: QUIETO
                    if self.agachado: # Si la IA decide quedarse quieta, se levanta
                        self.agachado = False
                        self.jugador.height = self.player_size[1]
                        self.jugador.bottom = self.ground_y
            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            # Aumentamos FPS para que todo el juego se sienta más rápido.
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()

