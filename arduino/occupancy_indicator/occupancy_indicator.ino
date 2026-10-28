/*
  Occupancy indicator for the Arduino UNO WiFi Rev2
  --------------------------------------------------
  Receives lines such as "COUNT:5" over USB serial from the Mac and shows
  the occupancy level with LEDs and a short warning beep.

    count <  LOW_THRESHOLD                     -> GREEN
    LOW_THRESHOLD <= count < MEDIUM_THRESHOLD  -> BLUE
    count >= MEDIUM_THRESHOLD                  -> RED
    count >= MAX_CAPACITY                      -> RED + one short beep
                                                  (again for each new person)
    no valid message for COMM_TIMEOUT_MS       -> all LEDs blink (lost link)

  Wiring (each LED needs its own 220-330 ohm resistor in series):
    pin 10 -> resistor -> RED   LED anode (long leg), cathode -> GND
    pin  9 -> resistor -> BLUE  LED anode,            cathode -> GND
    pin  8 -> resistor -> GREEN LED anode,            cathode -> GND
    pin  3 -> buzzer (+), buzzer (-) -> GND

  Keep these thresholds in sync with config.json on the Mac.
  No delay() is used: everything is timed with millis().
*/

// ------------------------------------------------------------ pins
const uint8_t RED_LED   = 10;
const uint8_t BLUE_LED  = 9;
const uint8_t GREEN_LED = 8;
const uint8_t BUZZER    = 3;

// ------------------------------------------------- settings (edit me)
const unsigned long BAUD_RATE         = 9600;   // must match config.json
const int  LOW_THRESHOLD              = 5;      // below this: green
const int  MEDIUM_THRESHOLD           = 10;     // at/above this: red
const int  MAX_CAPACITY               = 15;     // at/above this: red + beep
const unsigned long COMM_TIMEOUT_MS   = 6000;   // no valid message for this long = lost
const unsigned long LOST_BLINK_MS     = 500;    // blink period while link is lost
const unsigned long BEEP_DURATION_MS  = 200;    // length of the warning beep
const unsigned int  BEEP_FREQUENCY_HZ = 2000;   // used for a passive buzzer
const bool BUZZER_IS_PASSIVE          = true;   // false = active buzzer (plain on/off)
const unsigned long INVALID_PRINT_GAP_MS = 1000; // limit "invalid message" printouts

// ------------------------------------------------------------ state
const uint8_t BUFFER_SIZE = 16;
char lineBuffer[BUFFER_SIZE];
uint8_t lineLength = 0;
bool lineOverflow = false;

int  currentCount = -1;              // -1 = nothing received yet
bool haveValidMessage = false;
bool linkLost = true;                // start as "waiting for the Mac"
unsigned long lastValidMs = 0;
unsigned long lastInvalidPrintMs = 0;

bool beepActive = false;
unsigned long beepStartMs = 0;

unsigned long lastBlinkMs = 0;
bool blinkOn = false;

// ----------------------------------------------------------- helpers
void setLeds(bool red, bool blue, bool green) {
  digitalWrite(RED_LED,   red   ? HIGH : LOW);
  digitalWrite(BLUE_LED,  blue  ? HIGH : LOW);
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
}

void startBeep(unsigned long now) {
  beepActive = true;
  beepStartMs = now;
  if (BUZZER_IS_PASSIVE) {
    tone(BUZZER, BEEP_FREQUENCY_HZ);
  } else {
    digitalWrite(BUZZER, HIGH);
  }
}

void updateBuzzer(unsigned long now) {
  if (beepActive && (now - beepStartMs >= BEEP_DURATION_MS)) {
    beepActive = false;
    if (BUZZER_IS_PASSIVE) {
      noTone(BUZZER);
    } else {
      digitalWrite(BUZZER, LOW);
    }
  }
}

// Accepts exactly "COUNT:" followed by 1-4 digits.  Anything else is invalid.
// This matches parse_count_message() in src/serial_manager.py.
bool parseCount(const char *line, int &countOut) {
  const char prefix[] = "COUNT:";
  const uint8_t prefixLen = 6;
  if (strncmp(line, prefix, prefixLen) != 0) {
    return false;
  }
  const char *digits = line + prefixLen;
  size_t len = strlen(digits);
  if (len < 1 || len > 4) {
    return false;
  }
  int value = 0;
  for (size_t i = 0; i < len; i++) {
    if (digits[i] < '0' || digits[i] > '9') {
      return false;
    }
    value = value * 10 + (digits[i] - '0');
  }
  countOut = value;
  return true;
}

void handleLine(unsigned long now) {
  lineBuffer[lineLength] = '\0';
  int count;
  if (parseCount(lineBuffer, count)) {
    lastValidMs = now;                       // every valid line acts as a heartbeat
    haveValidMessage = true;
    if (count != currentCount) {
      int previous = currentCount;
      currentCount = count;
      Serial.print("Count received: ");
      Serial.println(currentCount);
      // Beep when someone arrives while at/over capacity.
      if (count >= MAX_CAPACITY && count > previous) {
        startBeep(now);
      }
    }
  } else if (now - lastInvalidPrintMs >= INVALID_PRINT_GAP_MS) {
    lastInvalidPrintMs = now;
    Serial.println("Ignored invalid message");
  }
}

// Reads whatever bytes are waiting, without ever blocking.
void readSerial(unsigned long now) {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      if (!lineOverflow && lineLength > 0) {
        handleLine(now);
      }
      lineLength = 0;
      lineOverflow = false;
    } else if (c == '\r') {
      // ignore carriage returns
    } else if (lineLength < BUFFER_SIZE - 1) {
      lineBuffer[lineLength++] = c;
    } else {
      lineOverflow = true;                   // too long: discard until newline
    }
  }
}

void updateLeds(unsigned long now) {
  bool lostNow = !haveValidMessage || (now - lastValidMs > COMM_TIMEOUT_MS);
  if (lostNow != linkLost) {
    linkLost = lostNow;
    Serial.println(linkLost ? "Communication lost" : "Communication OK");
  }

  if (linkLost) {
    // Blink all three LEDs together so it is obvious something is wrong.
    if (now - lastBlinkMs >= LOST_BLINK_MS) {
      lastBlinkMs = now;
      blinkOn = !blinkOn;
    }
    setLeds(blinkOn, blinkOn, blinkOn);
    return;
  }

  if (currentCount >= MEDIUM_THRESHOLD) {
    setLeds(true, false, false);             // red (also at maximum capacity)
  } else if (currentCount >= LOW_THRESHOLD) {
    setLeds(false, true, false);             // blue
  } else {
    setLeds(false, false, true);             // green
  }
}

// ------------------------------------------------------------ Arduino
void setup() {
  pinMode(RED_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  setLeds(false, false, false);
  digitalWrite(BUZZER, LOW);

  Serial.begin(BAUD_RATE);                   // do not wait for the Serial Monitor
  Serial.println("Occupancy indicator ready. Waiting for COUNT:<number>");
}

void loop() {
  unsigned long now = millis();
  readSerial(now);
  updateLeds(now);
  updateBuzzer(now);
}
