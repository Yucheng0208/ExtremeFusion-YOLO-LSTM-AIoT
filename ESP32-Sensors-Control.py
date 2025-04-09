# MicroPython version for ESP32
import network
import socket
import time
from machine import I2C, Pin
import esp32

# WiFi settings
SSID = 'YOUR_SSID'         # Replace with your WiFi SSID
PASSWORD = 'YOUR_PASSWORD' # Replace with your WiFi password
SERVER_IP = '192.168.1.100'  # Replace with your PC server IP
SERVER_PORT = 12345          # Replace with your PC server port

# Connect to WiFi
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        print('Connecting to WiFi...')
        time.sleep(1)
    print('WiFi connected:', wlan.ifconfig())

# Connect to TCP server
def connect_server():
    s = socket.socket()
    while True:
        try:
            s.connect((SERVER_IP, SERVER_PORT))
            print('Connected to server')
            return s
        except Exception as e:
            print('Connection failed, retrying...', e)
            time.sleep(2)

# Placeholder to simulate MAX30105 heart rate reading
def read_heart_rate():
    ir_value = esp32.hall_sensor()  # Fake reading
    heart_rate = ir_value / 10.0     # Scale for demo purposes
    return max(0, heart_rate)

# Placeholder to simulate MP34DT05 breathing sound detection
def read_breathing_volume():
    mic_value = esp32.raw_temperature()  # Fake reading
    volume = mic_value * 2               # Scale for demo purposes
    return max(0, volume)

# Main loop
def main():
    connect_wifi()
    s = connect_server()

    while True:
        try:
            heart_rate = read_heart_rate()
            volume = read_breathing_volume()
            message = "{:.1f},{}\n".format(heart_rate, int(volume))
            s.send(message.encode())
            print('Sent:', message.strip())
        except Exception as e:
            print('Error:', e)
            s.close()
            s = connect_server()
        time.sleep(0.5)  # Send every 500ms

if __name__ == '__main__':
    main()
