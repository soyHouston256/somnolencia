/*
 * =========================================================================
 *  Receptor de alarma de somnolencia  (brazalete vibrador)
 * =========================================================================
 *  Recibe por USB la senal que envia el script de Python:
 *      '1' -> PELIGRO (microsueno)  -> el motor vibra en patron
 *      '0' -> SEGURO                -> el motor se apaga
 *
 *  El patron NO usa delay(): se controla con millis(), asi el Arduino
 *  nunca deja de escuchar el puerto y no pierde el comando de apagado.
 *
 *  Conexion del modulo de vibracion:
 *      VCC -> 5V
 *      GND -> GND
 *      IN  -> pin 8
 *
 *  Serial.begin debe usar los mismos baudios que BAUDIOS en Python (9600).
 * =========================================================================
 */

const int  PIN_ALARMA = 8;      // pin IN del modulo de vibracion
const long BAUDIOS     = 9600;  // debe coincidir con el script de Python

const char SENAL_PELIGRO = '1';
const char SENAL_SEGURO  = '0';

// Patron de vibracion: cuanto vibra y cuanto pausa (en milisegundos).
const unsigned long VIBRA_MS = 250;
const unsigned long PAUSA_MS = 150;

bool          alarma_activa       = false;  // estamos en peligro?
bool          motor_encendido     = false;  // estado actual del motor
unsigned long marca_ultimo_cambio = 0;      // millis del ultimo cambio

void setup() {
  pinMode(PIN_ALARMA, OUTPUT);
  digitalWrite(PIN_ALARMA, LOW);
  Serial.begin(BAUDIOS);
}

void loop() {
  leer_senal();
  actualizar_vibracion();
}

// Lee el byte que manda Python y actualiza el estado de la alarma.
void leer_senal() {
  if (Serial.available() > 0) {
    char dato = Serial.read();
    if (dato == SENAL_PELIGRO) {
      alarma_activa = true;
    }
    if (dato == SENAL_SEGURO) {
      alarma_activa = false;
    }
  }
}

// Hace vibrar el motor en patron mientras la alarma este activa.
void actualizar_vibracion() {
  // Sin alarma: aseguramos el motor apagado y salimos.
  if (!alarma_activa) {
    if (motor_encendido) {
      motor_encendido = false;
      digitalWrite(PIN_ALARMA, LOW);
    }
    return;
  }

  // Con alarma: cuanto debe durar el estado actual (vibrando o en pausa).
  unsigned long duracion_actual;
  if (motor_encendido) {
    duracion_actual = VIBRA_MS;
  } else {
    duracion_actual = PAUSA_MS;
  }

  // Si ya paso ese tiempo, alternamos el estado del motor.
  unsigned long ahora = millis();
  if (ahora - marca_ultimo_cambio >= duracion_actual) {
    if (motor_encendido) {
      motor_encendido = false;
      digitalWrite(PIN_ALARMA, LOW);
    } else {
      motor_encendido = true;
      digitalWrite(PIN_ALARMA, HIGH);
    }
    marca_ultimo_cambio = ahora;
  }
}
