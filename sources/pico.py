"""

┌─────────────────────────────────────────┐
│              RASPBERRY PI 4             │
│                                         │
│  caméra → détection obstacles + ArUco   │
│  → calcul de navigation                 │
│  → commandes série vers Pico            │
│  → interface web mise à jour            │
└─────────────┬───────────────────────────┘
              │ USB série
              │ commandes : AVANCE / PIVOT_G / PIVOT_D / STOP / SCAN
              │ données   : C:15 L:30 R:45 TL:120 TR:118
              ▼
┌─────────────────────────────────────────┐
│              PICO (MicroPython)         │
│                                         │
│  moteurs + encodeurs + ultrasons        │
│  → exécute les commandes reçues         │
│  → envoie capteurs toutes les 100ms     │
└─────────────────────────────────────────┘

"""




from machine import Pin, PWM, time_pulse_us, UART
import time

# =========================
# UART vers Pi 4 (USB = uart0 natif via USB)
# On utilise sys.stdin/stdout car la Pico en mode USB CDC
# =========================
import sys
import select

# =========================
# MOTEURS
# =========================
IN1 = Pin(2, Pin.OUT)
IN2 = Pin(3, Pin.OUT)
ENA = PWM(Pin(4))
ENA.freq(1000)

IN3 = Pin(6, Pin.OUT)
IN4 = Pin(7, Pin.OUT)
ENB = PWM(Pin(5))
ENB.freq(1000)

# =========================
# ENCODEURS
# =========================
ENC_L_A = Pin(14, Pin.IN)
ENC_L_B = Pin(15, Pin.IN)
ENC_R_A = Pin(16, Pin.IN)
ENC_R_B = Pin(17, Pin.IN)

ticks_L = 0
ticks_R = 0

def callback_L(pin):
    global ticks_L
    ticks_L += 1 if ENC_L_B.value() else -1

def callback_R(pin):
    global ticks_R
    ticks_R += 1 if ENC_R_B.value() else -1

ENC_L_A.irq(trigger=Pin.IRQ_RISING, handler=callback_L)
ENC_R_A.irq(trigger=Pin.IRQ_RISING, handler=callback_R)

# =========================
# ULTRASONS x3
# =========================
TRIG_C = Pin(20, Pin.OUT); ECHO_C = Pin(21, Pin.IN)
TRIG_L = Pin(18, Pin.OUT); ECHO_L = Pin(19, Pin.IN)
TRIG_R = Pin(10, Pin.OUT); ECHO_R = Pin(11, Pin.IN)

def mesure(trig, echo):
    trig.value(0); time.sleep_us(2)
    trig.value(1); time.sleep_us(10)
    trig.value(0)
    d = time_pulse_us(echo, 1, 25000)
    return 999 if d < 0 else (d / 2) / 29.1

def lire_capteurs():
    dc = mesure(TRIG_C, ECHO_C); time.sleep_ms(20)
    dl = mesure(TRIG_L, ECHO_L); time.sleep_ms(20)
    dr = mesure(TRIG_R, ECHO_R); time.sleep_ms(20)
    return dc, dl, dr

# =========================
# MOTEURS
# =========================
BASE_SPEED = 32000
TURN_SPEED = 28000
BIAIS      = 5000  # correction biais gauche

def moteur_gauche(speed):
    if speed > 0:   IN1.value(1); IN2.value(0)
    elif speed < 0: IN1.value(0); IN2.value(1)
    else:           IN1.value(0); IN2.value(0)
    ENA.duty_u16(min(abs(int(speed)), 65535))

def moteur_droit(speed):
    if speed > 0:   IN3.value(1); IN4.value(0)
    elif speed < 0: IN3.value(0); IN4.value(1)
    else:           IN3.value(0); IN4.value(0)
    ENB.duty_u16(min(abs(int(speed)), 65535))

def stop():
    ENA.duty_u16(0); ENB.duty_u16(0)
    IN1.value(0); IN2.value(0)
    IN3.value(0); IN4.value(0)

def clamp(v): return max(min(v, 65535), -65535)

# =========================
# EXÉCUTION DES COMMANDES
# =========================
def executer(cmd):
    cmd = cmd.strip()
    if cmd == "AVANCE":
        moteur_gauche(clamp(BASE_SPEED + BIAIS))
        moteur_droit(clamp(BASE_SPEED))
    elif cmd == "RECUL":
        moteur_gauche(-TURN_SPEED)
        moteur_droit(-TURN_SPEED)
    elif cmd == "PIVOT_G":
        moteur_gauche(-TURN_SPEED)
        moteur_droit(TURN_SPEED)
    elif cmd == "PIVOT_D":
        moteur_gauche(TURN_SPEED)
        moteur_droit(-TURN_SPEED)
    elif cmd == "STOP":
        stop()
    elif cmd.startswith("VITESSE"):
        # format: VITESSE:L:R  ex: VITESSE:30000:35000
        parts = cmd.split(":")
        if len(parts) == 3:
            moteur_gauche(clamp(int(parts[1])))
            moteur_droit(clamp(int(parts[2])))

# =========================
# MAIN LOOP
# =========================
print("PICO_READY")
last_send = time.ticks_ms()

try:
    while True:
        # Lire commande si disponible
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline()
            if cmd:
                executer(cmd)

        # Envoyer données capteurs toutes les 100ms
        now = time.ticks_ms()
        if time.ticks_diff(now, last_send) >= 100:
            dc, dl, dr = lire_capteurs()
            print(f"DATA:C:{dc:.1f}:L:{dl:.1f}:R:{dr:.1f}:TL:{ticks_L}:TR:{ticks_R}")
            last_send = now

except KeyboardInterrupt:
    stop()