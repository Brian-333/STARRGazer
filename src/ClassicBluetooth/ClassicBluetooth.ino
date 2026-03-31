#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

#define LED_PIN 2   // Built-in LED on most ESP32 boards

volatile bool connected = false;

// 🔥 Callback function (runs on Bluetooth events)
void callback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param) {
  if (event == ESP_SPP_SRV_OPEN_EVT) {
    Serial.println("Client Connected!");
    connected = true;
  }

  if (event == ESP_SPP_CLOSE_EVT) {
    Serial.println("Client Disconnected!");
    connected = false;
  }
}

void bt_gap_callback(esp_bt_gap_cb_event_t event, esp_bt_gap_cb_param_t *param) {

  switch (event) {

    case ESP_BT_GAP_KEY_NOTIF_EVT:
      Serial.print("Passkey: ");
      Serial.println(param->key_notif.passkey);
      break;

    case ESP_BT_GAP_CFM_REQ_EVT:
      Serial.print("Confirm number: ");
      Serial.println(param->cfm_req.num_val);
      esp_bt_gap_ssp_confirm_reply(param->cfm_req.bda, true);
      break;

    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  // Register callback BEFORE begin()
  SerialBT.register_callback(callback);

  SerialBT.deleteAllBondedDevices();

  // 🔑 Set pairing PIN (IMPORTANT)
  SerialBT.enableSSP();      // Secure pairing

  if (!SerialBT.begin("ESP32_BT")) {
    Serial.println("Bluetooth failed!");
  } else {
    esp_bt_gap_register_callback(bt_gap_callback);
    Serial.println("Bluetooth ready. Pair now.");
  }
}

void loop() {
  connected = SerialBT.hasClient();
  if (connected) {
    // 🔁 Flash LED when connected
    digitalWrite(LED_PIN, HIGH);
    delay(300);
    digitalWrite(LED_PIN, LOW);
    delay(300);
  } else {
    // LED OFF when not connected
    digitalWrite(LED_PIN, LOW);
  }
}