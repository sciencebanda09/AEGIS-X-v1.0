#include <Servo.h>

// ── Pin definitions ──────────────────────────────────────────────────────────
#define PIN_SERVO_PAN     3
#define PIN_SERVO_TILT    5
#define PIN_SERVO_LAUNCH  6
#define PIN_LED_RED       7
#define PIN_LED_GREEN     8
#define PIN_MOTOR_L1      9
#define PIN_MOTOR_L2      10
#define PIN_MOTOR_R1      11
#define PIN_MOTOR_R2      12
#define PIN_BATT_SENSE    A0

// ── Servo home positions ─────────────────────────────────────────────────────
#define PAN_HOME      90     // degrees — facing forward
#define TILT_HOME     70     // degrees — angled slightly up
#define LAUNCH_ARMED  10     // degrees — trigger loaded
#define LAUNCH_FIRE   90     // degrees — trigger released / catapult

// ── Servo travel limits (physical safety) ────────────────────────────────────
#define PAN_MIN       5
#define PAN_MAX       175
#define TILT_MIN      20
#define TILT_MAX      130

// ── Launch parameters ────────────────────────────────────────────────────────
#define LAUNCH_HOLD_MS    120   // ms — hold trigger at fire angle
#define REARM_DELAY_MS    800   // ms — wait before re-arming trigger

// ── Scan parameters ──────────────────────────────────────────────────────────
#define SCAN_STEP_MS      40    // ms between scan steps
#define SCAN_STEP_DEG     3     // degrees per step

// ── Objects ──────────────────────────────────────────────────────────────────
Servo servoPan;
Servo servoTilt;
Servo servoLaunch;

// ── State ────────────────────────────────────────────────────────────────────
float  panAngle    = PAN_HOME;
float  tiltAngle   = TILT_HOME;
bool   scanning    = false;
int    scanDir     = 1;
unsigned long lastScanMs = 0;
bool   safeMode    = false;
int    launches    = 0;

String inputBuffer = "";

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  servoPan.attach(PIN_SERVO_PAN);
  servoTilt.attach(PIN_SERVO_TILT);
  servoLaunch.attach(PIN_SERVO_LAUNCH);

  pinMode(PIN_LED_RED,   OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_MOTOR_L1,  OUTPUT);
  pinMode(PIN_MOTOR_L2,  OUTPUT);
  pinMode(PIN_MOTOR_R1,  OUTPUT);
  pinMode(PIN_MOTOR_R2,  OUTPUT);

  // Go to home position
  cmdHome();

  // Status blink — 3 green flashes = ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED_GREEN, HIGH); delay(120);
    digitalWrite(PIN_LED_GREEN, LOW);  delay(100);
  }

  Serial.println("AEGIS-X ARDUINO READY");
  Serial.print("BATT ");
  Serial.println(readBattVolts(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // ── Parse serial input ────────────────────────────────────────────────────
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
      if (inputBuffer.length() > 64) inputBuffer = "";  // overflow guard
    }
  }

  // ── Scan loop ─────────────────────────────────────────────────────────────
  if (scanning && !safeMode) {
    unsigned long now = millis();
    if (now - lastScanMs >= SCAN_STEP_MS) {
      lastScanMs = now;
      panAngle += scanDir * SCAN_STEP_DEG;
      if (panAngle >= PAN_MAX || panAngle <= PAN_MIN) {
        scanDir = -scanDir;
      }
      panAngle = constrain(panAngle, PAN_MIN, PAN_MAX);
      servoPan.write((int)panAngle);
    }
  }

  // ── Battery report every 5 s ──────────────────────────────────────────────
  static unsigned long lastBattMs = 0;
  if (millis() - lastBattMs > 5000) {
    lastBattMs = millis();
    Serial.print("BATT ");
    Serial.println(readBattVolts(), 2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void processCommand(String cmd) {
  cmd.trim();
  String token = cmd.substring(0, cmd.indexOf(' ') < 0 ? cmd.length() : cmd.indexOf(' '));
  token.toUpperCase();

  if (token == "SERVO") {
    // SERVO <pan_deg> <tilt_deg>
    int sp1 = cmd.indexOf(' ');
    int sp2 = cmd.indexOf(' ', sp1 + 1);
    if (sp1 > 0 && sp2 > 0) {
      float pan  = cmd.substring(sp1 + 1, sp2).toFloat();
      float tilt = cmd.substring(sp2 + 1).toFloat();
      cmdServo(pan, tilt);
    } else {
      Serial.println("ERR SERVO_PARSE");
    }
  }
  else if (token == "LAUNCH") {
    cmdLaunch();
  }
  else if (token == "HOME") {
    cmdHome();
  }
  else if (token == "SCAN") {
    scanning = !safeMode;
    Serial.println("OK SCAN");
  }
  else if (token == "SAFE") {
    cmdSafe();
  }
  else if (token == "DRIVE") {
    // DRIVE <vx> <vy> <omega>
    int s1 = cmd.indexOf(' ');
    int s2 = cmd.indexOf(' ', s1 + 1);
    int s3 = cmd.indexOf(' ', s2 + 1);
    if (s1 > 0 && s2 > 0 && s3 > 0) {
      float vx    = cmd.substring(s1 + 1, s2).toFloat();
      float vy    = cmd.substring(s2 + 1, s3).toFloat();
      float omega = cmd.substring(s3 + 1).toFloat();
      cmdDrive(vx, vy, omega);
    } else {
      Serial.println("ERR DRIVE_PARSE");
    }
  }
  else {
    Serial.print("ERR UNKNOWN_CMD: "); Serial.println(cmd);
  }
}

// ── Command functions ─────────────────────────────────────────────────────────

void cmdServo(float pan, float tilt) {
  if (safeMode) { Serial.println("ERR SAFE_MODE"); return; }
  scanning  = false;  // stop scanning when actively aimed
  panAngle  = constrain(pan,  PAN_MIN,  PAN_MAX);
  tiltAngle = constrain(tilt, TILT_MIN, TILT_MAX);
  servoPan.write((int)panAngle);
  servoTilt.write((int)tiltAngle);
  // Echo actual position
  Serial.print("SERVO_ACK ");
  Serial.print(panAngle, 1);
  Serial.print(" ");
  Serial.println(tiltAngle, 1);
}

void cmdLaunch() {
  if (safeMode) { Serial.println("ERR SAFE_MODE"); return; }

  // Visual warning
  digitalWrite(PIN_LED_RED, HIGH);
  Serial.println("LAUNCH_START");

  // Fire trigger
  servoLaunch.write(LAUNCH_FIRE);
  delay(LAUNCH_HOLD_MS);

  // Re-arm
  servoLaunch.write(LAUNCH_ARMED);
  delay(REARM_DELAY_MS);

  launches++;
  digitalWrite(PIN_LED_RED, LOW);
  digitalWrite(PIN_LED_GREEN, HIGH);
  delay(100);
  digitalWrite(PIN_LED_GREEN, LOW);

  Serial.print("DONE LAUNCH#");
  Serial.println(launches);
}

void cmdHome() {
  safeMode  = false;
  scanning  = false;
  panAngle  = PAN_HOME;
  tiltAngle = TILT_HOME;
  servoPan.write(PAN_HOME);
  servoTilt.write(TILT_HOME);
  servoLaunch.write(LAUNCH_ARMED);
  stopMotors();
  Serial.println("OK HOME");
}

void cmdSafe() {
  safeMode = true;
  scanning = false;
  stopMotors();
  // Don't move servos abruptly — leave where they are
  Serial.println("OK SAFE");
}

// ── Drive (differential/mecanum chassis via L298N) ───────────────────────────
void cmdDrive(float vx, float vy, float omega) {
  if (safeMode) { Serial.println("ERR SAFE_MODE"); return; }

  // Differential drive mixing  (ignores vy for standard diff drive)
  // For mecanum, extend this to include vy
  float left  = constrain(vx - omega, -1.0, 1.0);
  float right = constrain(vx + omega, -1.0, 1.0);

  setMotor(PIN_MOTOR_L1, PIN_MOTOR_L2, left);
  setMotor(PIN_MOTOR_R1, PIN_MOTOR_R2, right);

  Serial.println("OK DRIVE");
}

void setMotor(int pinA, int pinB, float speed) {
  int pwm = (int)(abs(speed) * 255);
  pwm = constrain(pwm, 0, 255);
  if (speed > 0.05) {
    analogWrite(pinA, pwm);
    digitalWrite(pinB, LOW);
  } else if (speed < -0.05) {
    digitalWrite(pinA, LOW);
    analogWrite(pinB, pwm);
  } else {
    digitalWrite(pinA, LOW);
    digitalWrite(pinB, LOW);
  }
}

void stopMotors() {
  digitalWrite(PIN_MOTOR_L1, LOW); digitalWrite(PIN_MOTOR_L2, LOW);
  digitalWrite(PIN_MOTOR_R1, LOW); digitalWrite(PIN_MOTOR_R2, LOW);
}

// ── Battery voltage ───────────────────────────────────────────────────────────
float readBattVolts() {
  // Voltage divider: R1=10kΩ, R2=3.3kΩ → scale factor = (10+3.3)/3.3 = 4.03
  // Adjust VDIV_SCALE to match your actual resistor values
  const float VDIV_SCALE = 4.03;
  const float VREF       = 5.0;   // Arduino Uno Vcc
  int raw = analogRead(PIN_BATT_SENSE);
  return (raw / 1023.0) * VREF * VDIV_SCALE;
}
