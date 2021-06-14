import json
from enum import Enum
from typing import Tuple, Optional, Dict

import zmq

# Internal Globals (TODO: need to set these using an init method for GAEA)
DEBUG = False

# ZeroMQ State Variables
ZMQ_CONTEXT = zmq.Context()


class GodotConnectionType(Enum):
    """
    An enumeration for the supported Godot-AI-Bridge (GAB) connection types.
    """
    OBSERVATIONS = 1  # subscriber connection to the GAB observation publisher
    ACTIONS = 2  # client connection to the GAB action listener


class GodotConnection():
    """
    Encapsulates a single socket connection to a Godot-AI-Bridge (GAB) process associated with a Godot Environment.
    """

    DEFAULT_PROTOCOL = 'tcp'
    DEFAULT_HOSTNAME = 'localhost'
    DEFAULT_TOPIC = ''
    DEFAULT_RECONNECT = True

    RECV_MAX_TRIES = 5  # the number of attempts to receive a message from Godot
    SEND_MAX_TRIES = 5  # the number of attempts to send a message to Godot

    # these are used for server replies for previously sent requests
    RESPONSE_MAX_TRIES = 10  # the number of attempts to receive a reply
    RESPONSE_POLLING_TIMEOUT = 10  # the polling timeout (in ms)

    def __init__(self, port: int, type: GodotConnectionType, protocol: Optional[str] = DEFAULT_PROTOCOL,
                 hostname: Optional[str] = DEFAULT_HOSTNAME, topic: Optional[str] = DEFAULT_TOPIC,
                 options: Optional[Dict] = None, reconnect: Optional[bool] = DEFAULT_RECONNECT):
        """ Initializer

        :param port: the port number
        :param type: the connection type (see GodotConnectionType)
        :param protocol: the network protocol
        :param hostname: the hostname on which the Godot process is running
        :param topic: a message topic filter (e.g., "/agent/1")
        :param options: socket options (see PyZmq for a complete list)
        :param reconnect: indicates whether to automatically reconnect on errors
        """
        if options is None:
            options = []

        self.port = port
        self.type = type
        self.protocol = protocol
        self.hostname = hostname
        self.topic = topic
        self.options = options
        self.reconnect = reconnect

        # connection status (updated by open and close)
        self.is_connected = False

        # a ZMQ socket for this connection
        self._socket = None

    def get_connection_string(self) -> str:
        """ Creates a URI describing this connection.

        :return: a connection string to Godot-AI-Bridge (GAB)
        """
        return f'{self.protocol}://{self.hostname}:{self.port}'

    def open(self) -> None:
        """ Opens a socket connection to a Godot Environment """
        if self.is_connected:
            if self.reconnect:
                self.close()
            else:
                raise RuntimeError('Attempted to reconnect to Godot when reconnect is disabled!')

        if self.type == GodotConnectionType.ACTIONS:
            self._socket = self._establish_action_connection()
        elif self.type == GodotConnectionType.OBSERVATIONS:
            self._socket = self._establish_observation_connection()

        self.is_connected = True

    def close(self) -> None:
        """ Closes a previously opened connection to a Godot Environment """
        if self.is_connected:
            self._socket.close()

        self.is_connected = False

    def receive(self, max_tries: Optional[int] = RECV_MAX_TRIES) -> str:
        """ Receives a message from Godot. Messages are formatted as '<TOPIC> <JSON-ENCODED PAYLOAD>'.

        :param max_tries: the maximum number of receive attempts
        :return: a string containing the received message
        :raises RuntimeError: raised if unable to receive a message from Godot after max_tries attempts
        """
        wait_count = 0

        message = None
        while message is None and wait_count < max_tries:
            try:
                message = self._socket.recv_string()
                return message
            except zmq.error.Again:
                if DEBUG:
                    print(f'Godot was unavailable during receive. Retrying.', flush=True)

                wait_count += 1

        raise RuntimeError(f'Failed to receive message from Godot after {max_tries} attempts.')

    def send(self, message: Dict, max_tries: Optional[int] = SEND_MAX_TRIES) -> Dict:
        """ Sends a message to Godot. Returns Godot's reply indicating delivery success or failure.

        :param message: a dictionary containing the message details
        :param max_tries: the maximum number of send attempts
        :return: the reply from Godot indicating success or failure.
        """
        if DEBUG:
            print(f'sending message {message} to Godot.', flush=True)

        wait_count = 0
        while wait_count < max_tries:
            try:
                # sends the message JSON-encoded
                self._socket.send_string(json.dumps(message))
                return self._receive_response()
            except zmq.error.Again:
                if DEBUG:
                    print(f'Godot was unavailable during send. Retrying.', flush=True)

                wait_count += 1

        raise RuntimeError(f'Failed to send message to Godot after {max_tries} attempts.')

    def _receive_response(self, timeout: int = RESPONSE_POLLING_TIMEOUT, max_tries: int = RESPONSE_MAX_TRIES) -> Dict:
        """ Receives Godot's reply corresponding to a previously sent message.

        :param timeout: a timeout (in milliseconds) for polling Godot for the reply
        :param max_tries: the maximum number of receive attempts
        :return: the reply message as a dictionary
        :raises RuntimeError: raises an exception if unable to receive a reply from Godot after max_tries tries
        """
        wait_count = 0

        response = None
        while response is None and wait_count < max_tries:
            if self._socket.poll(timeout) & zmq.POLLIN != 0:
                response = self._socket.recv_json()
                return response
            else:
                if DEBUG:
                    print(f'Godot action listener failed to send a response. Retrying.', flush=True)

                wait_count += 1

        raise RuntimeError(f'Failed to receive response from action listener after {max_tries} attempts.')

    def _establish_action_connection(self) -> zmq.Socket:
        """ Establishes a socket connection to the Godot-AI-Bridge (GAB) action listener.

        :return: a ZMQ socket connection
        """
        socket = ZMQ_CONTEXT.socket(zmq.REQ)
        socket_options = {
            # configures timeouts - without a timeout on receive the process can hang indefinitely
            zmq.RCVTIMEO: 100,
            zmq.SNDTIMEO: 100,

            # configures the high watermark for outbound messages
            zmq.SNDHWM: 1,

            # pending messages are discarded immediately on socket close
            zmq.LINGER: 0
        }

        # merges default socket options with those supplied by user
        socket_options.update(self.options)
        self._set_options(socket, socket_options)

        connection_str = self.get_connection_string()
        socket.connect(connection_str)
        if DEBUG:
            print(f'connection established to Godot action listener @ {connection_str}!')

        return socket

    def _establish_observation_connection(self) -> zmq.Socket:
        """ Establishes a socket connection to the Godot-AI-Bridge (GAB) observation publisher.

        :return: a ZMQ socket connection
        """

        # establish subscriber connection
        socket = ZMQ_CONTEXT.socket(zmq.SUB)

        # filters messages by topic
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        socket_options = {
            # configures timeouts - without a timeout on receive the process can hang indefinitely
            zmq.RCVTIMEO: 100,

            # configures the high watermark for incoming messages
            zmq.RCVHWM: 1,

            # pending messages are discarded immediately on socket close
            zmq.LINGER: 0
        }

        # merges default socket options with those supplied by user
        socket_options.update(self.options)
        self._set_options(socket, socket_options)

        connection_str = self.get_connection_string()
        socket.connect(connection_str)
        if DEBUG:
            print(f'connection established to Godot observation publisher @ {connection_str}!')

        return socket

    def _set_options(self, socket: zmq.Socket, options: Dict) -> zmq.Socket:
        """ Configures the options for a ZMQ socket connection.

        :param socket: a ZMQ socket
        :param options: a dictionary containing the ZMQ option keys and their values (see PyZmq for the list of options)
        :return: the ZMQ socket with set options
        """
        for option, value in options.items():
            socket.setsockopt(option, value)
        return socket


# def wait_for_godot(condition: Callable, max_tries: int):
#     """ waits for a specific condition to be satisfied, or a max number of failed tries to occur.
#
#     :param condition:
#     :param max_tries:
#     :return:
#     """
#     pass


def parse_message(message: str) -> Tuple:
    """ Parses and unmarshals a JSON encoded message.

    :param message:
    :return: a Tuple containing the topic and message payload as an object.
    """
    topic, payload_enc = split_msg(message)
    payload = json.loads(payload_enc)

    return topic, payload


def split_msg(msg: str) -> Tuple:
    """ Splits message into topic and payload.

    :param msg: a ZeroMq message containing the topic string and JSON content
    :return: a tuple containing the topic and JSON encoded content
    """
    ndx = msg.find('{')
    return msg[0:ndx - 1], msg[ndx:]
