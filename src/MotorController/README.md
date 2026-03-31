# Motor Controller Set Up

## ESP-WROOM-32 Set Up

[ELEGOO ESP-WROOM-32 Setup](https://samueladesola.medium.com/how-to-set-up-esp32-wroom-32-b2100060470c)
1. Download Arduino IDE
2. Install esp32 board by espressif
3. Select the port and use board "DOIT ESP32 DEVKIT V1"

### Physical Pins

| ESP-32 Pin # | Motor Driver (Big) | Motor Driver (Small) |
|--------------|--------------------|----------------------|
| 25           | PUL+               |                      |
| 26           | DIR+               |                      |
| 27           | ENA+               |                      |
| 32           |                    | PUL+                 |
| 33           |                    | DIR+                 |
| 14           |                    | ENA+                 |
| GND          | PUL-               | PUL-                 |
| GND          | DIR-               | DIR-                 |
| GND          | ENA-               | ENA-                 |

All the GND above can be connected to the same GND pin

## Gear Ratio

