from collections import namedtuple
from enum import Enum
from typing import Callable, Tuple, Optional, Dict

import zmq
import json
import gym

# Internal Globals
DEBUG = False

# ZeroMQ State Variables
ZMQ_CONTEXT = zmq.Context()

# ZeroMQ Default GodotConnection Parameters
DEFAULT_RECV_TIMEOUT = 100  # in milliseconds
DEFAULT_SEND_TIMEOUT = 100

ACTION_TIMEOUT = 100
JOIN_TIMEOUT = 150
QUIT_TIMEOUT = 150


class GodotConnectionType(Enum):
    OBSERVATIONS = 1
    ACTIONS = 2


class GodotConnection():
    def __init__(self, port, type: GodotConnectionType, protocol: Optional[str] = 'tcp',
                 hostname: Optional[str] = 'localhost', topic: Optional[str] = '',
                 options=None, reconnect: Optional[bool] = True):
        """

        :param port:
        :param protocol:
        :param hostname:
        :param reconnect:
        """
        if options is None:
            options = []

        self.port = port
        self.type = type
        self.protocol = protocol
        self.hostname = hostname
        self.topic = topic
        self.options = options
        self.reconnect = True

        self._socket = None
        self._connected = False

    def get_connection_string(self):
        return f'{self.protocol}://{self.hostname}:{self.port}'

    def open(self):
        if self._connected and not self.reconnect:
            raise RuntimeError('Attempted to reconnect to Godot when reconnect option set to \'False\'!')

        if self.type == GodotConnectionType.ACTIONS:
            self._socket = self._establish_action_connection()
        elif self.type == GodotConnectionType.OBSERVATIONS:
            self._socket = self._establish_observation_connection()

        self._connected = True

    def close(self):

        self._connected = False

    def _establish_action_connection(self) -> zmq.Socket:
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

    def _set_options(self, socket, options):
        for option, value in options:
            socket.setsockopt(option, value)
        return socket


class GodotEnvWrapper(gym.Env):
    def __init__(self,
                 obs_conn: GodotConnection,
                 action_conn: GodotConnection,
                 agent_id: Optional[str] = '1',
                 max_steps: Optional[int] = float('inf'),
                 obs_parser: Optional[Callable] = lambda obs: [obs, None],
                 reward_func: Optional[Callable] = lambda obs: -1.0,
                 is_terminal: Optional[Callable] = lambda obs: False,
                 **kwargs):
        """

        :param obs_conn:
        :param action_conn:
        :param agent_id:
        :param max_steps:
        :param obs_parser:
        :param reward_func:
        :param is_terminal:
        :param kwargs:
        """

        self._agent_id = agent_id
        self._max_step = max_steps
        self._obs_parser = obs_parser
        self._reward_func = reward_func
        self._is_terminal = is_terminal
        self._kwargs = kwargs

        self.current_step = 0
        self.last_act_seq = 0
        self.last_obs = None
        self.done = True

    def _wait_for_action_obs(self, max_tries=100):
        wait_count = 0
        meta, obs, obs_seqno = self._receive_observation_from_godot()
        while obs_seqno is None or obs_seqno < self.last_act_seq:
            wait_count += 1
            if wait_count > max_tries:
                meta, obs = None, None
                break

            if DEBUG:
                print(f'agent {self._agent_id} waiting to receive obs with action_seqno {self.last_act_seq};'
                      + f'last received {obs_seqno}; wait count {wait_count}', flush=True)

            meta, obs, obs_seqno = self._receive_observation_from_godot()

        return meta, obs

    def step(self, action):
        if self.done:
            return RuntimeError('Environment is \'done\'. Call \'reset\' to continue.')

        self.current_step += 1

        meta, obs, info = None, None, None
        while obs is None:
            try:
                # TODO: Fix this. This needs to increment and set the last_action_seqno in the outgoing message.
                info, obs = self.act(action)

            except (AssertionError, RuntimeError, zmq.error.ZMQError) as e:
                print(f'agent {self._agent_id} received exception when sending action to server: {e}.', flush=True)
                print(f'agent {self._agent_id} attempting to recover by reconnecting to server')

                self._reconnect()
                self.last_obs = self.reset(clear_history=False)

        # remaining return values
        reward = self._reward_func(obs)
        done = self.check_done(obs)

        # this must be set after the call to _calculate_reward!
        self.last_obs = obs

        return obs, reward, done, info

    def reset(self, clear_history=True):
        """ respawns the agent within a running Godot environment and returns an initial observation """
        resetting = True
        while resetting:
            try:
                self.quit()
                self.join()

                if clear_history:
                    self.reset_history()

                # wait until observations start flowing for this agent id
                if DEBUG:
                    print(f'waiting for observations to arrive...', flush=True)

                _topic, payload = receive_message(max_tries=100)
                self.last_obs = self._obs_parser(payload)
                assert self.last_obs is not None, "assertion failed in reset: last observation is None!"

            except (AssertionError, RuntimeError, zmq.error.ZMQError) as e:
                print(f'agent {self._agent_id} received exception during reset: {e}.', flush=True)
                print(f'agent {self._agent_id} attempting to recover by reconnecting to server')

                self._reconnect()

            else:
                resetting = False

        return self.last_obs

    def render(self, mode='noop'):
        """ this is a noop. rendering is done in the Godot engine. """
        pass

    def act(self, action):
        self.last_act_seq += 1

        message = self._create_action_message(action)
        if DEBUG:
            print(f'agent {self._agent_id} sending action message {message} to Godot.')

        # send action to godot
        send_message(ZMQ_CONNECTIONS[ACTION_CONNECTION], message)

        # wait for response with corresponding observation

        # TODO: Fix this... how do we retrieve the last_action_seqno?
        info, obs = self._wait_for_action_obs()
        assert obs is not None, "assertion failed: post-action obs is None!"

        return info, obs

    def _create_action_message(self, action):
        header = {'type': 'action', 'id': self._agent_id, 'seqno': self.last_act_seq}
        data = {'action': action}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def join(self):
        pass

    def _create_join_message(self):
        header = {'type': 'join', 'id': self._agent_id}
        request = {'header': header, 'data': None}
        request_encoded = json.dumps(request)
        return request_encoded

    def quit(self):
        pass

    def _create_quit_message(self):
        header = {'type': 'quit', 'id': self._agent_id}
        request = {'header': header, 'data': None}
        request_encoded = json.dumps(request)
        return request_encoded

    def close(self):

        # attempt graceful agent exit
        quit()

        # explicitly release ZeroMQ socket connections
        for connection in ZMQ_CONNECTIONS.values():
            connection.close()

    def reset_history(self):
        self.current_step = 0
        self.last_obs = None
        self.last_act_seq = 0
        self.terminal_state = False

    def check_done(self, obs):
        # check: terminal state
        if self.terminal_state:
            return True

        # check: step based termination
        if (self._max_step - self.current_step) <= 0:
            return True

        # check: terminal state (based on user supplied callable)
        return self._is_terminal(obs)

    def _reconnect(self):
        print(f'agent {self._agent_id} is reconnecting to server!', flush=True)

        # close existing connection
        if DEBUG:
            print(f'agent {self._agent_id} is closing old connections!', flush=True)

        for connection_type, connection in self._connections.items():
            connection.close(linger=0)

        self._connections = {
            OBSERVATION_CONNECTION: self._establish_obs_conn(),
            ACTION_CONNECTION: self._establish_action_conn()
        }

    def _send_quit_to_godot(self):
        if DEBUG:
            print(f'agent {self._agent_id} is attempting to leave the world!', flush=True)

        message = self._create_quit_message()
        if not self._send_message_to_action_server(message, timeout=QUIT_TIMEOUT):
            raise RuntimeError(f'agent {self._agent_id} was unable to send quit messsage to Godot!')

        # wait until observations stop flowing for this agent id
        _, obs, _ = self._receive_observation_from_godot()
        while obs is not None:
            if DEBUG:
                print(f'agent {self._agent_id} left world, but Godot is still sending observations. '
                      + 'Waiting for them to stop...')
            _, obs, _ = self._receive_observation_from_godot()

        if DEBUG:
            print(f'agent {self._agent_id} is no longer receiving observations.')

        if DEBUG:
            print(f'agent {self._agent_id} has left the world!', flush=True)

    def _send_join_to_godot(self, max_tries=np.inf):
        if DEBUG:
            print(f'agent {self._agent_id} is attempting to join the world!', flush=True)

        message = self._create_join_message()
        if not self._send_message_to_action_server(message, timeout=JOIN_TIMEOUT):
            raise RuntimeError(f'agent {self._agent_id} was unable to send join messsage to Godot! aborting.')

        if DEBUG:
            print(f'agent {self._agent_id} has joined the world!', flush=True)


def wait_for_godot(condition: Callable, max_tries: int):
    """ waits for a specific condition to be satisfied, or a max number of failed tries to occur.

    :param condition:
    :param max_tries:
    :return:
    """
    pass


def receive_message(connection, as_json=False, max_tries=2) -> Optional[str]:
    wait_count = 0

    message = None
    while message is None and wait_count < max_tries:
        try:
            message = connection.recv_json() if as_json else connection.recv_string()
        except zmq.error.Again:
            if DEBUG:
                print(f'Godot was unavailable during receive. Retrying.', flush=True)

            wait_count += 1

    if wait_count >= max_tries:
        raise RuntimeError(f'Failed to receive message from Godot after {max_tries} attempts.')

    return message


def parse_message(message: str) -> Tuple:
    """ Parses and unmarshals a JSON encoded message.

    :param message:
    :return: a Tuple containing the topic and message payload as an object.
    """
    topic, payload_enc = split_msg(message)
    payload = json.loads(payload_enc)

    return topic, payload


def send_message(connection, message, max_tries=5):
    if DEBUG:
        print(f'sending message {message} to Godot.', flush=True)

    wait_count = 0
    while wait_count < max_tries:
        try:
            connection.send_string(message)
            return receive_response(connection, as_json=True)
        except zmq.error.Again:
            if DEBUG:
                print(f'Godot was unavailable during send. Retrying.', flush=True)

            wait_count += 1

    raise RuntimeError(f'Failed to send message to Godot after {max_tries} attempts.')


def receive_response(connection, as_json=False, max_tries=5):
    wait_count = 0

    response = None
    while response is None and wait_count < max_tries:
        if (connection.poll(DEFAULT_RECV_TIMEOUT)) & zmq.POLLIN != 0:
            response = connection.recv_json() if as_json else connection.recv_string()
        else:
            if DEBUG:
                print(f'Godot action listener failed to send a response. Retrying.', flush=True)

            wait_count += 1

    if wait_count >= max_tries:
        raise RuntimeError(f'Failed to receive response from action listener after {max_tries} attempts.')

    return response


def split_msg(msg: str) -> Tuple:
    """ Splits message into topic and payload.

    :param msg: a ZeroMq message containing the topic string and JSON content
    :return: a tuple containing the topic and JSON encoded content
    """
    ndx = msg.find('{')
    return msg[0:ndx - 1], msg[ndx:]
