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
import threading

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


# PICAR ORDERS
def order(value):
    if "SPEED" in value.upper():
        speed = value.split(':')
        bw.speed = int(speed[1])
        bw.forward()
    if value.upper() == 'L':
        print("left")
        fw.turn(int(90 - turning_angle))
    if value.upper() == 'R':
        print("right")
        fw.turn(int(90 + turning_angle))
    if value.upper() == 'S':
        print("straight")
        fw.turn(int(90))
    if value.upper() == 'LINE':
        print("Line")
        line_follower.main()
    if value.upper() == 'STOP':
        print("STOP")
        bw.speed = 0
        bw.forward()
    if value.upper() == 'FAST':
        print("FAST")
        bw.speed = 100
        bw.forward()
    if value.upper() == 'SLOW':
        print("right")
        bw.speed = 40
        bw.forward()
    if value.upper() == 'LIGHT':
        print("Siguiendo la luz")
        line_follower.stop()
        os.system('python /home/pi/Desktop/RPi_PiCar/SunFounder_PiCar-S/example/light_follower.py')
    if value.upper() == 'LIGHTULTRA':
        print("Siguiendo la luz")
        os.system('python /home/pi/Desktop/RPi_PiCar/SunFounder_PiCar-S/example/light_with_obsavoidance.py')
    if value.upper() == 'ULTRA':
        print("Esquivando objetos")
        # os.system('python /home/pi/Desktop/RPi_PiCar/SunFounder_PiCar-S/example/ultra_sonic_avoid.py')
        line_follower.stop()
    if value.upper() == 'STOP2':
        print("Realizando un Test")
        os.system('picar servo-install')
    if value.upper() == 'DISTANCE':
        print("DISTANCE:")
        try:
            # Publish a message
            distance = UA.get_distance()
            distancia = str(distance)
            print("DISTANCE:"+distancia)
            mqttc.publish(topic, " distancia: " + distancia + " ")
        except KeyboardInterrupt:
            print("error on MQTT")


# MQTT
# Define event callbacks
def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc))


def on_message(client, obj, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    order(str(msg.payload), )


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


class TxCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        Characteristic.__init__(self, bus, index, UART_TX_CHARACTERISTIC_UUID,
                                ['notify'], service)
        self.notifying = False
        GObject.io_add_watch(sys.stdin, GObject.IO_IN, self.on_console_input)

    def on_console_input(self, fd, condition):
        s = fd.readline()
        if s.isspace():
            pass
        else:
            self.send_tx(s)
        return True

    def send_tx(self, s):
        if not self.notifying:
            return
        value = []
        for c in s:
            value.append(dbus.Byte(c.encode()))
        self.PropertiesChanged(GATT_CHRC_IFACE, {'Value': value}, [])

    def StartNotify(self):
        if self.notifying:
            return
        self.notifying = True

    def StopNotify(self):
        if not self.notifying:
            return
        self.notifying = False


class RxCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        Characteristic.__init__(self, bus, index, UART_RX_CHARACTERISTIC_UUID,
                                ['write'], service)

    def WriteValue(self, value, options):
        comando = '{}'.format(bytearray(value).decode())
        distance = UA.get_distance()
        distancia = str(distance)
        print("command: " + comando + " distance: " + distancia + " ")

        order(comando)

        try:
            # Publish a message
            mqttc.publish(topic, "command: " + comando + " distance: " + distancia + " ")
        except KeyboardInterrupt:
            print("error on MQTT")


class UartService(Service):
    def __init__(self, bus, index):
        Service.__init__(self, bus, index, UART_SERVICE_UUID, True)
        self.add_characteristic(TxCharacteristic(bus, 0, self))
        self.add_characteristic(RxCharacteristic(bus, 1, self))


class Application(dbus.service.Object):
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
        return response


class UartApplication(Application):
    def __init__(self, bus):
        Application.__init__(self, bus)
        self.add_service(UartService(bus, 0))


class UartAdvertisement(Advertisement):
    def __init__(self, bus, index):
        Advertisement.__init__(self, bus, index, 'peripheral')
        self.add_service_uuid(UART_SERVICE_UUID)
        self.add_local_name(LOCAL_NAME)
        self.include_tx_power = True


def find_adapter(bus):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'),
                               DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    for o, props in objects.items():
        for iface in (LE_ADVERTISING_MANAGER_IFACE, GATT_MANAGER_IFACE):
            if iface not in props:
                continue
        return o
    return None


# -------------------------------------
# PICAR
# -------------------------------------

def straight_run():
    while True:
        bw.speed = 70
        bw.forward()
        fw.turn_straight()


def setup():
    if calibrate:
        cali()


def main():
    print('main')


def mainBLE():
    print('mainBLE')
    global mainloop
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    adapter = find_adapter(bus)
    if not adapter:
        print('BLE adapter not found')
        return

    service_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter),
        GATT_MANAGER_IFACE)
    ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                LE_ADVERTISING_MANAGER_IFACE)

    app = UartApplication(bus)
    adv = UartAdvertisement(bus, 0)

    mainloop = GObject.MainLoop()

    service_manager.RegisterApplication(app.get_path(), {},
                                        reply_handler=register_app_cb,
                                        error_handler=register_app_error_cb)
    ad_manager.RegisterAdvertisement(adv.get_path(), {},
                                     reply_handler=register_ad_cb,
                                     error_handler=register_ad_error_cb)

    # -------------------------------------
    # PICAR
    # -------------------------------------

    try:
        pmqtt = threading.Thread(target=mainMQTT)
        pmqtt.setDaemon(True)
        pmqtt.start()

        mainloop.run()
    except KeyboardInterrupt:
        adv.Release()


def mainMQTT():
    print("mainMQTT")
    rc = 0
    while rc == 0:
        time.sleep(1)
        rc = mqttc.loop()
    print("rc: " + str(rc))


if __name__ == '__main__':
    mainBLE()
