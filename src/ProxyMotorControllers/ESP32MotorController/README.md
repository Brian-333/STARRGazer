# ESP32 Motor Controller with Python Serial Controller

Python Code: Host
ESP32 or other microController: Proxy

## Protocol

Pre-req: Physical connection, Host and Proxy needs to be physically connect 

1. Initialization


## Payload

Host to Proxy:
(mode, payload)

| Name         | Mode | Payload | Description |
|--------------|------|---------|-------------|
| empty signal | 0    | 0       | Does nothing, an empty payload to keep connection alive. |
| init         | 1    | "pan_step_pin, pan_dir_pin, tilt_step_pin, tilt_step_pin" | Initialize pin numbers to use on the proxy |
| move         | 2    | "target_pan, 
