
const int EMG_PIN = 34;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);
}

void loop() {
  int value = analogRead(EMG_PIN);
  Serial.println(value);

  delayMicroseconds(2000);  // 目标约 500 Hz
}
