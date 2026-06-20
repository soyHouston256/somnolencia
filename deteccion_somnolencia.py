"""
=============================================================================
 Detección de somnolencia del conductor
 -> prevención de accidente vial (un conductor se queda dormido al volante)
=============================================================================
 Caso real: sobre el Anillo Periférico un conductor se quedó dormido, se salió
 de su carril y se estrelló contra el muro de contención. La CAUSA RAÍZ es la
 somnolencia (un microsueño). Este sistema ataca esa causa: vigila el rostro
 del conductor y AVISA antes de que ocurra el choque.

 Cómo funciona, en 4 pasos:

   1) MALLA FACIAL  -> MediaPipe Face Mesh ubica los puntos del rostro.
   2) SEÑAL DE OJOS -> con los puntos del ojo se calcula el EAR (Eye Aspect
                       Ratio). Al cerrar el ojo, el EAR baja.
   3) MICROSUEÑO    -> si el EAR queda por debajo del umbral durante varios
                       cuadros seguidos = ojos cerrados de verdad (no un
                       parpadeo).
   4) RESPUESTA     -> en vez de saltar de "despierto" a "dormido", se integra
                       un INDICE DE SOMNOLENCIA continuo (0-100): sube de forma
                       gradual mientras el ojo permanece cerrado (ponderado por
                       cuánto se cierra) y por los bostezos, y baja cuando el
                       conductor vuelve a estar alerta. El indice define un
                       estado escalonado y sensible:
                         ALERTA (verde) -> SOMNOLENCIA LEVE (amarillo) ->
                         FATIGA (naranja) -> SOMNOLENCIA ALTA (naranja-rojo) ->
                         MICROSUEÑO (rojo).
                       Asi la transicion es suave y se avisa mucho antes. En un
                       auto real, el rojo dispararía una alarma sonora.

 El EAR y el MAR son COCIENTES de distancias, así que el sistema es
 independiente de la resolución. Además se autocalibra: mide el EAR base de
 cada persona en los primeros cuadros (con los ojos abiertos) y de ahí deriva
 el umbral, en lugar de fijar un número arbitrario.

 Dependencias: OpenCV, NumPy y MediaPipe.
     pip install opencv-python numpy mediapipe

 Uso:
     python deteccion_somnolencia.py                  # usa la webcam
     python deteccion_somnolencia.py --video clip.mp4 # usa un vide
     python deteccion_somnolencia.py --sin-ventana    # solo guarda el MP4

 Durante la vista en vivo, ESC o 'q' cierran la ventana.

 Integrantes: Max Houston Ramírez Martel, Yesseliz Choque Becerra,
              Meliza Sosa Mariño, Alberto Velásquez Santos
 Visión Computacional | UTEC Posgrado | Prof. Royer Rojas Malasquez
=============================================================================
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# ==========================================================================
# CONFIGURACIÓN  (todo lo ajustable, en un solo lugar y con nombre propio)
# ==========================================================================

# --- Entrada / salida ---
CAMARA_POR_DEFECTO = 0                    # índice de la webcam
ARCHIVO_SALIDA     = "resultado_somnolencia.mp4"
FPS_POR_DEFECTO    = 20.0                 # se usa si la fuente no reporta fps
ESPEJO_WEBCAM      = True                 # mostrar la webcam en espejo (natural)

# --- Hardware (actuador opcional: un Arduino por puerto serie) ---
BAUDIOS               = 9600             # debe coincidir con el sketch del Arduino
BYTE_PELIGRO          = b"1"             # se envía al ENTRAR en microsueño
BYTE_SEGURO           = b"0"             # se envía al VOLVER a un estado seguro
TIEMPO_ESPERA_ARDUINO = 2.0             # el Arduino se reinicia al abrir el puerto

# --- Puntos de la malla facial (índices de MediaPipe Face Mesh) ---
# Cada ojo se describe con 6 puntos en el orden [P1, P2, P3, P4, P5, P6]:
#   P1, P4 -> esquinas del ojo (ancho)
#   P2, P3 -> párpado superior   |   P5, P6 -> párpado inferior
OJO_DERECHO   = (33, 160, 158, 133, 153, 144)
OJO_IZQUIERDO = (362, 385, 387, 263, 373, 380)

# Boca: dos esquinas (ancho) y dos puntos centrales (apertura vertical).
BOCA_IZQUIERDA = 61
BOCA_DERECHA   = 291
BOCA_SUPERIOR  = 13
BOCA_INFERIOR  = 14

# --- Umbrales de decisión ---
FACTOR_OJO_CERRADO   = 0.75   # umbral EAR = este factor * EAR base de la persona
EAR_BASE_DE_RESPALDO = 0.25   # EAR base si la autocalibración no logra medir
FRAMES_MICROSUENO    = 12     # cuadros seguidos con el ojo cerrado = microsueño
UMBRAL_BOSTEZO       = 0.60   # MAR por encima de esto = boca muy abierta
FRAMES_BOSTEZO       = 10     # cuadros seguidos de boca abierta = bostezo

# --- Índice de somnolencia (0-100): transición gradual, no un salto binario ---
# En vez de decidir "despierto / dormido", se integra una señal continua que
# sube mientras el ojo se cierra y baja cuando vuelve a abrirse. Así el sistema
# es mucho más sensible y avisa por niveles antes de llegar al microsueño.
ALFA_SUAVIZADO_EAR = 0.30   # peso del EAR nuevo en la media móvil exponencial (EMA)
FACTOR_PERCLOS     = 0.80   # umbral P80: ojo "cerrado" si apertura < 80% del EAR base (estándar PERCLOS)
VENTANA_PERCLOS_S  = 4.0    # segundos de la ventana deslizante de PERCLOS
SUBIDA_INDICE      = 3.5    # cuánto sube el índice por cuadro con el ojo completamente cerrado
BAJADA_INDICE      = 2.0    # cuánto baja el índice por cuadro cuando el conductor está alerta
APORTE_BOSTEZO     = 1.5    # puntos extra por cuadro durante un bostezo

# Cortes del índice para el estado escalonado (orden de severidad creciente).
UMBRAL_LEVE       = 25      # índice >= -> SOMNOLENCIA LEVE
UMBRAL_FATIGA     = 50      # índice >= -> FATIGA
UMBRAL_ALTA       = 75      # índice >= -> SOMNOLENCIA ALTA
UMBRAL_MICROSUENO = 90      # índice >= -> MICROSUENO (dispara la alarma)

# --- Autocalibración del EAR base ---
FRAMES_CALIBRACION = 30       # primeros cuadros con rostro (ojos abiertos)

# --- Visualización ---
ALTO_BANNER         = 60      # franja de estado, arriba
ALTO_PANEL          = 95      # panel de métricas, abajo (incluye barra de nivel)
OPACIDAD_PANEL      = 0.45
GROSOR_BORDE        = 12
RADIO_PUNTO         = 2
ESCALA_BANNER       = 0.85
ESCALA_PANEL        = 0.60
UMBRAL_BRILLO_TEXTO = 140     # sobre este brillo se usa texto negro, si no blanco

# Colores en BGR (OpenCV NO usa RGB).
COLOR_VERDE       = (0, 180, 0)
COLOR_AMARILLO    = (0, 200, 220)
COLOR_NARANJA     = (0, 140, 255)
COLOR_NARANJA_ROJO = (0, 80, 255)
COLOR_ROJO        = (0, 0, 255)
COLOR_BLANCO      = (255, 255, 255)
COLOR_NEGRO       = (0, 0, 0)

# Estados posibles del conductor (orden de severidad creciente).
ESTADO_ALERTA     = "ALERTA"
ESTADO_LEVE       = "SOMNOLENCIA LEVE"
ESTADO_FATIGA     = "FATIGA"
ESTADO_ALTA       = "SOMNOLENCIA ALTA"
ESTADO_MICROSUENO = "MICROSUENO"

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
CODEC  = cv2.VideoWriter_fourcc("m", "p", "4", "v")


# ==========================================================================
# MÉTRICAS GEOMÉTRICAS DEL ROSTRO  (Pasos 2 y 4)
# ==========================================================================

def distancia(punto_a, punto_b):
    """Distancia euclidiana entre dos puntos (x, y)."""
    return float(np.linalg.norm(np.array(punto_a) - np.array(punto_b)))


def calcular_ear(puntos_ojo):
    """
    Eye Aspect Ratio de un ojo descrito por 6 puntos [P1..P6]:

        EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)

    Es alto con el ojo abierto y cae hacia 0 al cerrarse. Al ser un cociente
    de distancias, no depende de la resolución ni del tamaño de la cara.
    """
    p1, p2, p3, p4, p5, p6 = puntos_ojo
    vertical = distancia(p2, p6) + distancia(p3, p5)
    horizontal = distancia(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def calcular_mar(superior, inferior, izquierda, derecha):
    """
    Mouth Aspect Ratio: apertura vertical de la boca sobre su ancho.
    Sube cuando la boca se abre (bostezo).
    """
    vertical = distancia(superior, inferior)
    horizontal = distancia(izquierda, derecha)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def punto_en_pixeles(landmarks, indice, ancho, alto):
    """Convierte un landmark normalizado (0-1) de MediaPipe a píxeles (x, y)."""
    punto = landmarks[indice]
    return (int(punto.x * ancho), int(punto.y * alto))


def puntos_en_pixeles(landmarks, indices, ancho, alto):
    """Igual que punto_en_pixeles, pero para una lista de índices."""
    return [punto_en_pixeles(landmarks, i, ancho, alto) for i in indices]


# ==========================================================================
# DECISIÓN DEL ESTADO  (Paso 4: máquina de estados escalonada)
# ==========================================================================

def color_de_estado(estado):
    """Color BGR asociado a cada estado (de verde a rojo según severidad)."""
    if estado == ESTADO_MICROSUENO:
        return COLOR_ROJO
    if estado == ESTADO_ALTA:
        return COLOR_NARANJA_ROJO
    if estado == ESTADO_FATIGA:
        return COLOR_NARANJA
    if estado == ESTADO_LEVE:
        return COLOR_AMARILLO
    return COLOR_VERDE


def mensaje_de_estado(estado):
    """Texto del banner para cada estado (solo ASCII para que se vea bien)."""
    if estado == ESTADO_MICROSUENO:
        return "MICROSUENO - DESPIERTA"
    if estado == ESTADO_ALTA:
        return "SOMNOLENCIA ALTA - DETENTE PRONTO"
    if estado == ESTADO_FATIGA:
        return "FATIGA - TOMA UN DESCANSO"
    if estado == ESTADO_LEVE:
        return "SOMNOLENCIA LEVE - ATENCION"
    return "CONDUCTOR ALERTA"


def actualizar_indice_somnolencia(indice_actual, ear_suave, ear_base, hay_bostezo):
    """
    Integra el índice de somnolencia continuo (0-100) cuadro a cuadro.

    En lugar de decidir "ojo abierto / cerrado", mide CUÁNTO está cerrado y
    acumula esa evidencia: el índice sube de forma proporcional al cierre y
    baja cuando el conductor está alerta, logrando una transición gradual.
    """
    base = ear_base if ear_base > 0 else EAR_BASE_DE_RESPALDO
    apertura_relativa = ear_suave / base
    apertura_relativa = max(0.0, min(1.0, apertura_relativa))

    # nivel_cierre en [0, 1]: 0 = ojo bien abierto, 1 = ojo completamente cerrado.
    nivel_cierre = (FACTOR_PERCLOS - apertura_relativa) / FACTOR_PERCLOS
    nivel_cierre = max(0.0, min(1.0, nivel_cierre))

    indice = indice_actual
    if nivel_cierre > 0:
        indice += SUBIDA_INDICE * nivel_cierre
    else:
        indice -= BAJADA_INDICE

    if hay_bostezo:
        indice += APORTE_BOSTEZO

    return max(0.0, min(100.0, indice))


def estado_desde_indice(indice):
    """Traduce el índice de somnolencia (0-100) al estado escalonado."""
    if indice >= UMBRAL_MICROSUENO:
        return ESTADO_MICROSUENO
    if indice >= UMBRAL_ALTA:
        return ESTADO_ALTA
    if indice >= UMBRAL_FATIGA:
        return ESTADO_FATIGA
    if indice >= UMBRAL_LEVE:
        return ESTADO_LEVE
    return ESTADO_ALERTA


def color_texto_sobre(color_fondo):
    """Negro sobre fondos claros, blanco sobre fondos oscuros (legibilidad)."""
    azul, verde, rojo = color_fondo
    brillo = 0.114 * azul + 0.587 * verde + 0.299 * rojo
    if brillo > UMBRAL_BRILLO_TEXTO:
        return COLOR_NEGRO
    return COLOR_BLANCO


# ==========================================================================
# DIBUJO SOBRE EL CUADRO
# ==========================================================================

def dibujar_ojos_y_boca(frame, pts_ojo_der, pts_ojo_izq, pts_boca, color):
    """Resalta los puntos que usa el algoritmo: contorno de ojos y boca."""
    for puntos in (pts_ojo_der, pts_ojo_izq):
        contorno = np.array(puntos, dtype=np.int32)
        cv2.polylines(frame, [contorno], True, color, 1)
        for x, y in puntos:
            cv2.circle(frame, (x, y), RADIO_PUNTO, color, -1)
    for x, y in pts_boca:
        cv2.circle(frame, (x, y), RADIO_PUNTO, color, -1)


def dibujar_borde_estado(frame, estado, hay_rostro):
    """Marco de color según el estado: comunica el nivel de un vistazo."""
    if hay_rostro:
        color = color_de_estado(estado)
    else:
        color = COLOR_BLANCO
    alto, ancho = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (ancho - 1, alto - 1), color, GROSOR_BORDE)


def dibujar_banner(frame, estado, calibrando, hay_rostro):
    """Franja superior con el mensaje principal."""
    ancho = frame.shape[1]
    if not hay_rostro:
        color = COLOR_BLANCO
        texto = "BUSCANDO ROSTRO"
    elif calibrando:
        color = COLOR_VERDE
        texto = "CALIBRANDO - MANTEN LOS OJOS ABIERTOS"
    else:
        color = color_de_estado(estado)
        texto = mensaje_de_estado(estado)
    cv2.rectangle(frame, (0, 0), (ancho, ALTO_BANNER), color, -1)
    cv2.putText(frame, texto, (15, 40), FUENTE, ESCALA_BANNER,
                color_texto_sobre(color), 2)


def dibujar_panel(frame, ear, mar, ear_base, perclos, indice, estado,
                  hay_rostro, calibrando):
    """Panel inferior translúcido con las métricas en vivo y la barra de nivel."""
    alto, ancho = frame.shape[:2]
    y0 = alto - ALTO_PANEL

    capa = frame.copy()
    cv2.rectangle(capa, (0, y0), (ancho, alto), COLOR_NEGRO, -1)
    cv2.addWeighted(capa, OPACIDAD_PANEL, frame, 1 - OPACIDAD_PANEL, 0, frame)

    if not hay_rostro:
        cv2.putText(frame, "Sin rostro en cuadro", (15, y0 + 45),
                    FUENTE, ESCALA_PANEL, COLOR_BLANCO, 2)
        return

    umbral_ear = FACTOR_OJO_CERRADO * ear_base
    estado_calib = "calibrando" if calibrando else "listo"
    linea1 = "EAR {:.2f}  (umbral {:.2f}, base {:.2f})  PERCLOS {:.0f}%".format(
        ear, umbral_ear, ear_base, perclos)
    linea2 = "MAR {:.2f}  |  calibracion: {}".format(mar, estado_calib)
    cv2.putText(frame, linea1, (15, y0 + 22), FUENTE, ESCALA_PANEL, COLOR_BLANCO, 2)
    cv2.putText(frame, linea2, (15, y0 + 44), FUENTE, ESCALA_PANEL, COLOR_BLANCO, 2)

    # Barra horizontal del índice de somnolencia (0-100).
    etiqueta = "Somnolencia: {:.0f}/100".format(indice)
    cv2.putText(frame, etiqueta, (15, y0 + 66), FUENTE, ESCALA_PANEL,
                COLOR_BLANCO, 2)
    barra_x0 = 15
    barra_x1 = ancho - 15
    barra_y0 = y0 + 74
    barra_y1 = y0 + 88
    cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x1, barra_y1),
                  COLOR_BLANCO, 1)
    ancho_util = barra_x1 - barra_x0 - 2
    relleno = int(ancho_util * max(0.0, min(100.0, indice)) / 100.0)
    if relleno > 0:
        cv2.rectangle(frame, (barra_x0 + 1, barra_y0 + 1),
                      (barra_x0 + 1 + relleno, barra_y1 - 1),
                      color_de_estado(estado), -1)


# ==========================================================================
# ACTUADOR DE HARDWARE  (opcional: envía la señal de alarma a un Arduino)
# ==========================================================================

def abrir_actuador(puerto):
    """
    Abre el puerto serie del Arduino y devuelve el objeto serie, o None.
    El sistema funciona igual sin hardware: si no hay puerto o falla la
    conexión, se devuelve None y simplemente no se envían señales (la demo
    nunca se cae por falta de placa).
    """
    if puerto is None:
        return None
    try:
        import serial  # pyserial; solo se necesita si hay hardware conectado
    except ImportError:
        print("Aviso: instala 'pyserial' para usar el actuador "
              "(pip install pyserial). Sigo sin hardware.")
        return None
    try:
        actuador = serial.Serial(puerto, BAUDIOS, timeout=1)
        time.sleep(TIEMPO_ESPERA_ARDUINO)  # el Arduino se reinicia al conectarse
        print("Actuador conectado en {}.".format(puerto))
        return actuador
    except Exception as error:
        print("Aviso: no se pudo abrir {} ({}). Sigo sin hardware.".format(
            puerto, error))
        return None


def enviar_senal(actuador, en_peligro):
    """Envía '1' (peligro) o '0' (seguro) al Arduino. Sin hardware, no hace nada."""
    if actuador is None:
        return
    if en_peligro:
        actuador.write(BYTE_PELIGRO)
    else:
        actuador.write(BYTE_SEGURO)


def cerrar_actuador(actuador):
    """Apaga la alarma y cierra el puerto serie al terminar."""
    if actuador is None:
        return
    enviar_senal(actuador, False)
    actuador.close()


# ==========================================================================
# ARGUMENTOS Y FUENTE DE VIDEO
# ==========================================================================

def leer_argumentos():
    """Lee la configuración desde la línea de comandos con argparse."""
    parser = argparse.ArgumentParser(
        description="Detección de somnolencia del conductor (prevención de choque)."
    )
    parser.add_argument(
        "--video", default=None,
        help="Ruta de un video. Si se omite, se usa la webcam.")
    parser.add_argument(
        "--camara", type=int, default=CAMARA_POR_DEFECTO,
        help="Índice de la webcam (por defecto 0).")
    parser.add_argument(
        "--sin-ventana", action="store_true",
        help="No mostrar la ventana en vivo; solo guardar el video de salida.")
    parser.add_argument(
        "--puerto", default=None,
        help="Puerto serie del Arduino (ej. COM3 o /dev/ttyACM0). "
             "Si se omite, no se usa hardware.")
    return parser.parse_args()


def abrir_fuente(args):
    """Abre la webcam o el video y devuelve la captura, su descripción y si es webcam."""
    es_webcam = args.video is None
    if es_webcam:
        captura = cv2.VideoCapture(args.camara)
        descripcion = "webcam #{}".format(args.camara)
    else:
        captura = cv2.VideoCapture(args.video)
        descripcion = "video '{}'".format(args.video)
    if not captura.isOpened():
        raise SystemExit("No se pudo abrir la fuente: {}.".format(descripcion))
    return captura, descripcion, es_webcam


# ==========================================================================
# PROGRAMA PRINCIPAL
# ==========================================================================

def main():
    args = leer_argumentos()
    captura, descripcion, es_webcam = abrir_fuente(args)

    # --- Paso 1: dimensiones de la fuente y preparación de la salida ---
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FPS_POR_DEFECTO

    escritor = cv2.VideoWriter(ARCHIVO_SALIDA, CODEC, fps, (ancho, alto))
    actuador = abrir_actuador(args.puerto)

    malla_facial = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --- Estado que persiste entre cuadros ---
    valores_calibracion = []
    ear_base = EAR_BASE_DE_RESPALDO
    cuadros_ojo_cerrado = 0
    cuadros_boca_abierta = 0
    cuadros_totales = 0
    cuadros_en_alarma = 0
    primer_microsueno_s = None
    peligro_anterior = False

    # Estado del índice de somnolencia continuo.
    ear_suave = None                  # EAR suavizado (EMA); None hasta el primer rostro
    indice_somnolencia = 0.0          # señal integrada 0-100
    indice_maximo = 0.0               # pico alcanzado (para el resumen)
    ventana_perclos = deque(maxlen=max(1, int(VENTANA_PERCLOS_S * fps)))
    perclos = 0.0                     # % de tiempo con el ojo cerrado en la ventana

    print("Procesando {} ({}x{} @ {:.0f} fps)...".format(
        descripcion, ancho, alto, fps))

    # --- Paso 1 a 4: un solo recorrido sobre los cuadros ---
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        if es_webcam and ESPEJO_WEBCAM:
            frame = cv2.flip(frame, 1)

        cuadros_totales += 1
        tiempo_s = cuadros_totales / fps

        # Paso 1: malla facial (MediaPipe trabaja en RGB).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = malla_facial.process(frame_rgb)

        hay_rostro = resultado.multi_face_landmarks is not None
        calibrando = len(valores_calibracion) < FRAMES_CALIBRACION
        ear = 0.0
        mar = 0.0
        estado = ESTADO_ALERTA

        if hay_rostro:
            landmarks = resultado.multi_face_landmarks[0].landmark

            # Paso 2: señal de ojos (EAR), promedio de ambos ojos.
            pts_ojo_der = puntos_en_pixeles(landmarks, OJO_DERECHO, ancho, alto)
            pts_ojo_izq = puntos_en_pixeles(landmarks, OJO_IZQUIERDO, ancho, alto)
            ear = (calcular_ear(pts_ojo_der) + calcular_ear(pts_ojo_izq)) / 2.0

            # Paso 4 (señal secundaria): apertura de la boca (MAR).
            sup = punto_en_pixeles(landmarks, BOCA_SUPERIOR, ancho, alto)
            inf = punto_en_pixeles(landmarks, BOCA_INFERIOR, ancho, alto)
            izq = punto_en_pixeles(landmarks, BOCA_IZQUIERDA, ancho, alto)
            der = punto_en_pixeles(landmarks, BOCA_DERECHA, ancho, alto)
            mar = calcular_mar(sup, inf, izq, der)

            # Paso 3: autocalibración del EAR base (mediana, robusta a parpadeos).
            if calibrando:
                valores_calibracion.append(ear)
                ear_base = float(np.median(valores_calibracion))
            umbral_ear = FACTOR_OJO_CERRADO * ear_base

            # Paso 3: conteo de cuadros seguidos (no un parpadeo suelto).
            if ear < umbral_ear:
                cuadros_ojo_cerrado += 1
            else:
                cuadros_ojo_cerrado = 0

            if mar > UMBRAL_BOSTEZO:
                cuadros_boca_abierta += 1
            else:
                cuadros_boca_abierta = 0

            hay_bostezo = cuadros_boca_abierta >= FRAMES_BOSTEZO

            # Paso 4: índice de somnolencia continuo.
            # EAR suavizado (EMA) para que el índice no salte con el ruido.
            if ear_suave is None:
                ear_suave = ear
            else:
                ear_suave = ALFA_SUAVIZADO_EAR * ear + (1 - ALFA_SUAVIZADO_EAR) * ear_suave

            # PERCLOS: % de cuadros recientes con el ojo "cerrado" (apertura < P80).
            base = ear_base if ear_base > 0 else EAR_BASE_DE_RESPALDO
            apertura_relativa = ear_suave / base
            ventana_perclos.append(1 if apertura_relativa < FACTOR_PERCLOS else 0)
            perclos = 100.0 * sum(ventana_perclos) / len(ventana_perclos)

            if calibrando:
                # Durante la calibración no acumulamos: el índice queda en 0.
                indice_somnolencia = 0.0
                estado = ESTADO_ALERTA
            else:
                indice_somnolencia = actualizar_indice_somnolencia(
                    indice_somnolencia, ear_suave, ear_base, hay_bostezo)
                estado = estado_desde_indice(indice_somnolencia)

            indice_maximo = max(indice_maximo, indice_somnolencia)

            dibujar_ojos_y_boca(frame, pts_ojo_der, pts_ojo_izq,
                                [sup, inf, izq, der], color_de_estado(estado))
        else:
            # Sin rostro: el índice decae gradualmente y reiniciamos conteos.
            cuadros_ojo_cerrado = 0
            cuadros_boca_abierta = 0
            ear_suave = None
            indice_somnolencia = max(0.0, indice_somnolencia - BAJADA_INDICE)

        # Registro del evento de microsueño.
        if estado == ESTADO_MICROSUENO:
            cuadros_en_alarma += 1
            if primer_microsueno_s is None:
                primer_microsueno_s = tiempo_s

        # Señal al hardware: solo en el CAMBIO de estado (no saturar el puerto).
        en_peligro = estado == ESTADO_MICROSUENO
        if en_peligro != peligro_anterior:
            enviar_senal(actuador, en_peligro)
            peligro_anterior = en_peligro

        # Paso 4: anotación final del cuadro.
        dibujar_borde_estado(frame, estado, hay_rostro)
        dibujar_banner(frame, estado, calibrando, hay_rostro)
        dibujar_panel(frame, ear, mar, ear_base, perclos, indice_somnolencia,
                      estado, hay_rostro, calibrando)

        escritor.write(frame)

        if not args.sin_ventana:
            cv2.imshow("Deteccion de somnolencia", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27 or tecla == ord("q"):
                break

    captura.release()
    escritor.release()
    malla_facial.close()
    cerrar_actuador(actuador)
    cv2.destroyAllWindows()

    # --- Resumen ---
    print("\nResumen")
    print("  Cuadros procesados : {}".format(cuadros_totales))
    print("  EAR base calibrado : {:.3f}".format(ear_base))
    print("  Indice maximo      : {:.0f}/100".format(indice_maximo))
    print("  Cuadros en alarma  : {}".format(cuadros_en_alarma))
    if primer_microsueno_s is not None:
        print("  Primer microsueno  : t = {:.2f} s".format(primer_microsueno_s))
    else:
        print("  Primer microsueno  : no se detecto")
    print("  Video guardado en  : {}".format(ARCHIVO_SALIDA))


if __name__ == "__main__":
    main()
