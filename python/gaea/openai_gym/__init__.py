from typing import Optional, Callable

import gym
import gaea


class GodotEnvWrapper(gym.Env):
    def __init__(self,
                 obs_conn: gaea.GodotConnection,
                 action_conn: gaea.GodotConnection,
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
