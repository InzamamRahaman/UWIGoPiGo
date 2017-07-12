import easygopigo3 as easy
import gopigo3 as go
from __future__ import print_function
from six.moves import input
import time


ROBOT = easy.EasyGoPiGo3()
MAX_INT = 10000
SENSOR = None
SERVO = None
SLEEP_TIME = 0.3
PORT = 'I2C1'

def init_sensor():
    """
    Initializes servo
    :return:
    """
    global ROBOT
    global SENSOR
    # if the sensor has not been initialised yet
    if not SENSOR:
        SENSOR = easy.DistanceSensor(PORT, ROBOT)

def measure():
    """
    Uses the distance sensor to measure the distance from the robot to the nearest object
    with the sensor's scope
    :return: the distance to the nearest object in centimetres
    """
    global SENSOR
    init_sensor()
    distance = SENSOR.read()
    return distance

def init_servo():
    """
    Initializes servo
    :return:
    """
    global SERVO
    global ROBOT
    if not SERVO:
        SERVO = ROBOT.init_servo("SERVO1")

def rotate_servo(position):
    """
    Rotates servo to a specific position
    :param position: the postiion to rotate towards
    :return: None
    """
    global SERVO
    init_servo()
    SERVO.rotate_servo(position)


def reset_servo():
    """
    Resets servo to default position
    :return: None
    """
    global SERVO
    init_servo()
    SERVO.reset_servo()

def forward():
    """
    Moves the robot forward
    :return: None
    """
    global ROBOT
    ROBOT.forward()

def backward():
    """
    Moves the robot backwards
    :return: None
    """
    global ROBOT
    ROBOT.backward()




def get_robot():
    """
    Grabs the robot object (breaks encapsulation :( )
    :return: None
    """
    global ROBOT
    return ROBOT

def go_forward(distance):
    """
    Makes the robot go forward a specified distance
    :param distance: the distance to travel in centimetres
    :return: None
    """
    global ROBOT
    ROBOT.drive_cm(distance)


def go_backwards(distance):
    """
    Makes the robot go backwards a specified distance
    :param distance: the distance to travel in centimetres
    :return: None
    """
    global ROBOT
    ROBOT.drive_cm(-1 * distance)


def rotate_right(angle=90):
    """
    Makes the robot rotate in a right direction by a specified angle
    :param angle: the angle to rotate by in degrees
    :return: None
    """
    global ROBOT
    ROBOT.turn_degrees(angle)



def rotate_left(angle=90):
    """
    Makes the robot rotate in a left direction by a specified angle
    :param angle: the angle to rotate by in degrees
    :return: None
    """
    global ROBOT
    ROBOT.turn_degrees(-1 * angle)


def set_speed(speed):
    """
    Changes the current speed of the robot
    :param speed: the new speed
    :return: None
    """
    global ROBOT
    ROBOT.set_speed(speed)


def get_speed():
    """
    Gets the speed from the robot
    :return: the current speed of the robot
    """
    global ROBOT
    ROBOT.get_speed()


def march_forward(braking_distance=MAX_INT):
    """
    Causes the robot to move forwards until it reaches with a specified distance of an arbitray object
    :param braking_distance: the minimum distance between the robot and object (optional: default ignore obstacles)
    :return: None
    """
    global ROBOT
    distance = measure()

    while distance < braking_distance:
        ROBOT.forward()
        time.sleep(SLEEP_TIME)
        distance = measure()


def stop():
    """
    Stops the robot
    :return: None
    """
    global ROBOT
    ROBOT.stop()


def reverse():
    """
    Reverses the robot
    :return: None
    """
    global ROBOT
    ROBOT.backward()

