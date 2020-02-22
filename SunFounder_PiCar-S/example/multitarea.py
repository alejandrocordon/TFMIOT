from threading import Thread

import os
import sys
import threading
from datetime import date

import dbus
import dbus.mainloop.glib
from example_advertisement import Advertisement
from example_advertisement import register_ad_cb, register_ad_error_cb
from example_gatt_server import Service, Characteristic
from example_gatt_server import register_app_cb, register_app_error_cb
from gi.repository import GObject

import line_follower

from SunFounder_Line_Follower import Line_Follower
from SunFounder_Ultrasonic_Avoidance import Ultrasonic_Avoidance
from picar import front_wheels
from picar import back_wheels
import time
import picar

import paho.mqtt.client as mqtt


# BLE init
BLUEZ_SERVICE_NAME = 'org.bluez'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
GATT_CHRC_IFACE = 'org.bluez.GattCharacteristic1'
UART_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
UART_RX_CHARACTERISTIC_UUID = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'
UART_TX_CHARACTERISTIC_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'
LOCAL_NAME = 'RaspberryPi3_UART'
mainloop = None

# PICAR
print("INIT")
picar.setup()

REFERENCES = [200, 200, 200, 200, 200]
# calibrate = True
calibrate = False
forward_speed = 80
backward_speed = 70
turning_angle = 40
dooms_day = False

max_off_track_count = 40

delay = 0.0005

fw = front_wheels.Front_Wheels(db='config')
bw = back_wheels.Back_Wheels(db='config')
lf = Line_Follower.Line_Follower()

lf.references = REFERENCES
fw.ready()
bw.ready()
fw.turning_max = 45

# Picar Ultrasonic init
UA = Ultrasonic_Avoidance.Ultrasonic_Avoidance(20)


# MQTT
# Define event callbacks
def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc))


def on_message(client, obj, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))


def on_publish(client, obj, mid):
    print("mid: " + str(mid))


def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log(client, obj, level, string):
    print(string)



mqttc = mqtt.Client()
# Assign event callbacks
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe

# Connect
mqttc.username_pw_set('ibnyofaw', 'UDbgKs77-wUN')
mqttc.connect('hairdresser.cloudmqtt.com', '18849')

topic = 'masteriot'
# Start subscribe, with QoS level 0
mqttc.subscribe(topic, 0)



def mainBLE():
    while True:
        time.sleep(1)
        print('A\n')


def mainMQTT():
    while True:
        rc = mqttc.loop()
    print("rc: " + str(rc))


if __name__ == "__main__":
    t1 = Thread(target=mainBLE)
    t2 = Thread(target=mainMQTT)

    t1.setDaemon(True)
    t2.setDaemon(True)
    t1.start()
    t2.start()

    while True:
        pass
