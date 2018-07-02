import easygopigo3 as easy
import gopigo3 as go
from __future__ import print_function
from six.moves import input
import time

#Tensorflow stuff
from __future__ import absolute_import
from __future__ import division
import os.path
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf


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

#Tensorflow code
FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
# MODIFICATION BY SAM ABRAHAMS
tf.app.flags.DEFINE_integer('warmup_runs', 0,
                            "Number of times to run Session before starting test")
tf.app.flags.DEFINE_integer('num_runs', 1,
                            "Number of sample runs to collect benchmark statistics")
# END OF MODIFICATION

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    start_time = time.time()
    create_graph()
    graph_time = time.time() - start_time

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        # MODIFICATION BY SAM ABRAHAMS
        for i in range(FLAGS.warmup_runs):
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})

        runs = []
        for i in range(FLAGS.num_runs):
            start_time = time.time()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            runs.append(time.time() - start_time)

        predictions = np.squeeze(predictions)

        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-1:][::-1]
        results = []
        for node_id in top_k:
            print(node_id)
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            # print('%s (score = %.5f)' % (human_string, score))
            results.append(human_string)
        '''
        for i, run in enumerate(runs):
          print('Run %03d:\t%0.4f seconds' % (i, run))
        print('---')
        print('Best run: %0.4f' % min(runs))
        print('Worst run: %0.4f' % max(runs))
        print('Average run: %0.4f' % float(sum(runs) / len(runs)))
        print('Build graph time: %0.4f' % graph_time)
        print('Number of warmup runs: %d' % FLAGS.warmup_runs)
        print('Number of test runs: %d' % FLAGS.num_runs)
        # END OF MODIFICATION
        '''
        return results, score

def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        print("Downloading to ",filepath)
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                 reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def init_eyes():
    # FLAGS.model_dir = 'inception'
    print("Opening eyes...")
    maybe_download_and_extract()
    print("Eyes opened!")

def see(image='/tmp/tmp.jpeg'):
    return run_inference_on_image(image)
