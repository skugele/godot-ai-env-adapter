import sys

import zmq
import numpy as np
import json
import gym

from stable_baselines.common.env_checker import check_env

DEFAULT_RECV_TIMEOUT = 100  # in milliseconds
DEFAULT_SEND_TIMEOUT = 100

ACTION_TIMEOUT = 100
JOIN_TIMEOUT = 150
QUIT_TIMEOUT = 150

# TODO: Set this from an observation from Godot
DIM_OBSERVATIONS = 6  # SMELL -> 1 & 2, SOMATOSENSORY -> 3, TOUCH -> 4, VELOCITY -> 5 & 6
# DIM_OBSERVATIONS = 2 # SMELL -> 1 & 2
DIM_ACTIONS = 4

# Keys for connection dictionary
CONN_KEY_OBSERVATIONS = 'OBS'
CONN_KEY_ACTIONS = 'ACTIONS'


# TODO: Move this
def split(m):
    """ Splits message into separate topic and content strings.
    :param m: a ZeroMq message containing the topic string and JSON content
    :return: a tuple containing the topic and JSON content
    """
    ndx = m.find('{')
    return m[0:ndx - 1], m[ndx:]


# OpenAI documentation on creating custom environments:
#         https://github.com/openai/gym/blob/master/docs/creating-environments.md

# TODO: Split this off into another base class that this one inherits from that contains more general Godot concerns,
#       like connection management
class SimpleAnimatWorld(gym.Env):
    _args = None

    # Godot environment connections (using ZeroMQ)
    _connections = None

    def __init__(self, agent_id, obs_port, action_port, args=None):
        self._agent_id = agent_id
        self._obs_port = obs_port
        self._action_port = action_port
        self._args = args

        self._curr_step = 0
        self._max_step = self._args.max_steps_per_episode
        self._action_seqno = 0

        self._last_obs = None

        self.action_space = gym.spaces.Discrete(2 ** DIM_ACTIONS - 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(DIM_OBSERVATIONS,))

        # ZeroMQ connection context - shared by all network sockets
        self._context = zmq.Context()

        # establish connections (and fail fast if Godot process not running)
        self._connections = {
            CONN_KEY_OBSERVATIONS: self._establish_obs_conn(),
            CONN_KEY_ACTIONS: self._establish_action_conn()
        }

        self._terminal_state_reached = True

    def _wait_for_action_obs(self, max_tries=100):
        wait_count = 0
        meta, obs, obs_seqno = self._receive_observation_from_godot()
        while obs_seqno is None or obs_seqno < self._action_seqno:
            wait_count += 1
            if wait_count > max_tries:
                meta, obs = None, None
                break

            if self._args.debug:
                print(f'agent {self._agent_id} waiting to receive obs with action_seqno {self._action_seqno};'
                      + f'last received {obs_seqno}; wait count {wait_count}', flush=True)

            meta, obs, obs_seqno = self._receive_observation_from_godot()

        return meta, obs

    def step(self, action):
        if self._check_done():
            print('error: attempting to step a \'done\' environment!', flush=True)
            return

        self._curr_step += 1

        meta, obs = None, None
        while obs is None:
            try:
                action_sent = self._send_action_to_godot(action)
                assert action_sent, "assertion failed: unable to send action!"

                if self._args.debug:
                    print(f'agent {self._agent_id} sent action {action} to Godot.')

                # wait for corresponding observation
                meta, obs = self._wait_for_action_obs()
                assert obs is not None, "assertion failed: post-action obs is None!"

            except (AssertionError, RuntimeError, zmq.error.ZMQError) as e:
                print(f'agent {self._agent_id} received exception when sending action to server: {e}.', flush=True)
                print(f'agent {self._agent_id} attempting to recover by reconnecting to server')

                self._reconnect()
                self._last_obs = self.reset(clear_history=False)

        # remaining return values
        reward = self._calculate_reward(obs)
        done = self._check_done()
        info = {'godot_info': meta}  # metadata about agent's observations

        if self._args.debug:
            print(f'agent {self._agent_id} -> last_obs: {self._last_obs}\nobs: {obs}\nreward: {reward}\ndone: {done}\n')
            print(f'agent {self._agent_id} -> info: {info}')

        # this must be set after the call to _calculate_reward!
        self._last_obs = obs

        return obs, reward, done, info

    def reset(self, clear_history=True):
        """ respawns the agent within a running Godot environment and returns an initial observation """
        resetting = True
        while resetting:
            try:
                # Godot ignores quits for non-existent agent ids. So always send it in case of stale connections.
                self._send_quit_to_godot()
                self._send_join_to_godot()

                if clear_history:
                    self._reset_history()

                # wait until observations start flowing for this agent id
                if self._args.debug:
                    print(f'agent {self._agent_id} is waiting for observations to arrive...', flush=True)

                _, self._last_obs, _ = self._receive_observation_from_godot(max_tries=100)
                assert self._last_obs is not None, "assertion failed: last obs is None!"

            except (AssertionError, RuntimeError, zmq.error.ZMQError) as e:
                print(f'agent {self._agent_id} received exception during reset: {e}.', flush=True)
                print(f'agent {self._agent_id} attempting to recover by reconnecting to server')

                self._reconnect()

            else:
                resetting = False

        return self._last_obs

    def render(self, mode='noop'):
        """ this is a noop. rendering is done in the Godot engine. """
        pass

    def close(self):
        self._send_quit_to_godot()

        # explicitly release ZeroMQ socket connections
        for conn in self._connections.values():
            conn.close()

    def verify(self):
        """ perform sanity checks on the environment """
        check_env(self, warn=True)

    def _reset_history(self):
        self._curr_step = 0
        self._last_obs = None
        self._action_seqno = 0
        self._terminal_state_reached = False

    def _check_done(self):
        # check: terminal state
        if self._terminal_state_reached:
            return True

        # check: step based termination
        if (self._max_step - self._curr_step) <= 0:
            return True

        return False

    def _establish_action_conn(self):
        socket = self._context.socket(zmq.REQ)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(self._action_port)

        if self._args.debug:
            print('establishing Godot action client using URL ', conn_str)

        socket.connect(conn_str)

        # configure timeout - without a timeout on receive the process can hang indefinitely
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_RECV_TIMEOUT)
        socket.setsockopt(zmq.SNDTIMEO, DEFAULT_SEND_TIMEOUT)

        socket.setsockopt(zmq.SNDHWM, 1)

        # pending messages are discarded immediately on socket close
        socket.setsockopt(zmq.LINGER, 0)

        return socket

    def _get_topic(self):
        return f'/agents/{self._agent_id}'

    def _establish_obs_conn(self):
        # establish subscriber connection
        socket = self._context.socket(zmq.SUB)

        # filters messages by topic
        socket.setsockopt_string(zmq.SUBSCRIBE, self._get_topic())

        # configure timeout
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_RECV_TIMEOUT)
        socket.setsockopt(zmq.RCVHWM, 1)

        # pending messages are discarded immediately on socket close
        socket.setsockopt(zmq.LINGER, 0)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(self._obs_port)

        if self._args.debug:
            print('establishing Godot sensors subscriber using URL ', conn_str)

        socket.connect(conn_str)

        return socket

    def _create_action_message(self, action):
        header = {'type': 'action', 'id': self._agent_id, 'seqno': self._action_seqno}
        data = {'action': int(action)}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _create_quit_message(self):
        header = {'type': 'quit', 'id': self._agent_id}
        data = {}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _create_join_message(self):
        header = {'type': 'join', 'id': self._agent_id}
        data = {}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _send(self, connection, message, max_tries=5):
        wait_count = 0

        while True:
            # REMOVE THIS
            if wait_count > 0:
                print(f'agent {self._agent_id} is spinning in _send! (wait count: {wait_count})')

            # TODO: Should this be changed to a polling method?
            try:
                connection.send_string(message)
            except zmq.error.Again:
                if self._args.debug:
                    print(f'Received EAGAIN: Godot was unavailable during send. Retrying.', flush=True)

                wait_count += 1

                if wait_count > max_tries:
                    raise RuntimeError(f'Failed to send message {message}')
            else:
                if self._args.debug:
                    print(f'agent {self._agent_id} sent message {message} to Godot.', flush=True)
                break

    def _receive_response(self, connection, as_json=False, timeout=DEFAULT_RECV_TIMEOUT, max_tries=5):
        wait_count = 0

        message = None
        while message is None and wait_count < max_tries:
            # REMOVE THIS
            if wait_count > 0:
                print(f'agent {self._agent_id} is spinning in _receive_response! (wait count: {wait_count})')

            if (connection.poll(timeout)) & zmq.POLLIN != 0:
                message = connection.recv_json() if as_json else connection.recv_string()
            else:
                wait_count += 1
                if self._args.debug:
                    print(f'waiting on a response from Godot. current wait count: {wait_count}', flush=True)

        return message

    def _reconnect(self):
        print(f'agent {self._agent_id} is reconnecting to server!', flush=True)

        # close existing connection
        if self._args.debug:
            print(f'agent {self._agent_id} is closing old connections!', flush=True)

        for connection_type, connection in self._connections.items():
            connection.close(linger=0)

        self._connections = {
            CONN_KEY_OBSERVATIONS: self._establish_obs_conn(),
            CONN_KEY_ACTIONS: self._establish_action_conn()
        }

    def _receive(self, connection, as_json=False, max_tries=2):
        wait_count = 0

        message = None
        while message is None and wait_count < max_tries:
            # REMOVE THIS
            if wait_count > 0:
                print(f'agent {self._agent_id} is spinning in _receive! (wait count: {wait_count})')

            try:
                message = connection.recv_json() if as_json else connection.recv_string()
            except zmq.error.Again:
                if self._args.debug:
                    print(f'agent {self._agent_id} received EAGAIN: Godot was unavailable during receive. Retrying.',
                          flush=True)

                wait_count += 1
                if self._args.debug:
                    print(f'agent {self._agent_id} waiting on a response from Godot. current wait count: {wait_count}',
                          flush=True)

        return message

    def _send_message_to_action_server(self, message, timeout=DEFAULT_SEND_TIMEOUT):
        connection = self._connections[CONN_KEY_ACTIONS]

        if self._args.debug:
            print(f'agent {self._agent_id} sending message {message} to Godot.', flush=True)

        self._send(connection, message)
        server_reply = self._receive_response(connection, timeout=timeout, as_json=True)
        if not server_reply:
            raise RuntimeError(f'agent {self._agent_id} failed to receive a reply from Godot for request {message}.')

        return server_reply is not None

    def _send_action_to_godot(self, action):
        self._action_seqno += 1
        if isinstance(action, np.ndarray):
            action = action.tolist()

        message = self._create_action_message(action)
        return self._send_message_to_action_server(message, timeout=ACTION_TIMEOUT)

    def _send_quit_to_godot(self):
        if self._args.debug:
            print(f'agent {self._agent_id} is attempting to leave the world!', flush=True)

        message = self._create_quit_message()
        if not self._send_message_to_action_server(message, timeout=QUIT_TIMEOUT):
            raise RuntimeError(f'agent {self._agent_id} was unable to send quit messsage to Godot!')

        # wait until observations stop flowing for this agent id
        _, obs, _ = self._receive_observation_from_godot()
        while obs is not None:
            if self._args.debug:
                print(f'agent {self._agent_id} left world, but Godot is still sending observations. '
                      + 'Waiting for them to stop...')
            _, obs, _ = self._receive_observation_from_godot()

        if self._args.debug:
            print(f'agent {self._agent_id} is no longer receiving observations.')

        if self._args.debug:
            print(f'agent {self._agent_id} has left the world!', flush=True)

    def _send_join_to_godot(self, max_tries=np.inf):
        if self._args.debug:
            print(f'agent {self._agent_id} is attempting to join the world!', flush=True)

        message = self._create_join_message()
        if not self._send_message_to_action_server(message, timeout=JOIN_TIMEOUT):
            raise RuntimeError(f'agent {self._agent_id} was unable to send join messsage to Godot! aborting.')

        if self._args.debug:
            print(f'agent {self._agent_id} has joined the world!', flush=True)

    def _receive_observation_from_godot(self, max_tries=1):
        connection = self._connections[CONN_KEY_OBSERVATIONS]

        header, obs, action_seqno = None, None, None

        # receive observation message (encoded as TOPIC + [SPACE] + json_encoded(PAYLOAD))
        message = self._receive(connection, max_tries=max_tries)
        if message:
            _, payload_enc = split(message)
            payload = json.loads(payload_enc)
            header, data = payload['header'], payload['data']

            obs = self._parse_observation(data)
            action_seqno = data['LAST_ACTION']

        return header, obs, action_seqno

    def _parse_observation(self, data):
        obs = None
        try:
            data_as_list = data['SMELL'] + [data['SOMATOSENSORY']] + [data['TOUCH']] + data['VELOCITY']

            obs = np.round(np.array(data_as_list), decimals=6)
        except KeyError as e:
            print(f'exception {e} occurred in observation message {data}', flush=True)

        return obs

    def _calculate_reward(self, obs):
        assert obs is not None, "assertion failed: obs is None in calculate_reward!"
        assert self._last_obs is not None, "assertion failed: last_obs is None in calculate_reward!"

        new_smell_intensity = obs[0] + obs[1]
        new_satiety = obs[2]
        new_touch = obs[3]

        old_smell_intensity = self._last_obs[0] + self._last_obs[1]
        old_satiety = self._last_obs[2]
        old_touch = self._last_obs[3]

        # health = np.concatenate(obs, axis=0)[:, 4]

        # satiety_multiplier = 10000
        # health_multiplier = 10
        # smell_multiplier = 100

        smell_delta = new_smell_intensity - old_smell_intensity
        # print(f'smell delta: {smell_delta}')

        # TODO: This should be non-linear (use a sigmoid function or something of that nature)
        satiety_delta = new_satiety - old_satiety
        # health_delta = health[-1] - health[0]

        reward = 0.0
        reward += 10 * satiety_delta

        # reward stronger smells if did not eat (the check is to prevent reward deductions due to
        # consumed food causing a strong negative smell_delta)
        reward += 10 * smell_delta if satiety_delta <= 0 else 0.0

        # agent starved. end of episode.
        if new_satiety == 0:
            reward -= 500
            self._terminal_state_reached = True

        return reward
