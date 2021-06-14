import json
import unittest

from gaea import GodotConnection, GodotConnectionType


class TestGodotConnection(unittest.TestCase):
    def test_initializer(self):

        try:
            port = 10001
            type = GodotConnectionType.OBSERVATIONS
            protocol = 'tcp'
            hostname = 'hostname'
            topic = 'topic'
            options = {'invalid option': 86}
            reconnect = False

            # Test 1: Check defaults (all optional other than port and type)
            conn = GodotConnection(port=port, type=type)

            self.assertEqual(conn.protocol, conn.DEFAULT_PROTOCOL, 'incorrect default for protocol')
            self.assertEqual(conn.hostname, conn.DEFAULT_HOSTNAME, 'incorrect default for hostname')
            self.assertEqual(conn.topic, conn.DEFAULT_TOPIC, 'incorrect default for topic')
            self.assertEqual(conn.reconnect, conn.DEFAULT_RECONNECT, 'incorrect default for reconnect')
            self.assertEqual(conn.options, [], 'incorrect default for options')

            # Test 2: Check that initializer parameters match object's property values
            conn = GodotConnection(port=port, type=type, protocol=protocol, hostname=hostname, topic=topic,
                                   options=options, reconnect=reconnect)
            self.assertEqual(conn.port, port, "port mismatch after initializer")
            self.assertEqual(conn.hostname, hostname, "hostname mismatch after initializer")
            self.assertEqual(conn.topic, topic, "topic mismatch after initializer")
            self.assertEqual(conn.reconnect, reconnect, "reconnect mismatch after initializer")
            self.assertDictEqual(conn.options, options, "options mismatch after initializer")

            # Test 3: Verify is_connected is False
            self.assertEqual(conn.is_connected, False)

        except Exception as e:
            self.fail(f'unexpected exception: {e}')

    def test_connection_string(self):
        try:
            conn = GodotConnection(port=10001, type=GodotConnectionType.OBSERVATIONS)
            conn_str = conn.get_connection_string()
            self.assertEqual(conn_str, f'{conn.DEFAULT_PROTOCOL}://{conn.DEFAULT_HOSTNAME}:10001')
        except Exception as e:
            self.fail(f'unexpected exception: {e}')

    # TODO: this needs to be modified to use mocks for the underlying sockets
    def test_open_and_close_connection(self):

        # test successful connections
        for port, type in zip([10001, 10002], list(GodotConnectionType)):
            try:
                conn = GodotConnection(port=port, type=GodotConnectionType.OBSERVATIONS)

                # Test 1: Verify is_connected is True after open
                conn.open()
                self.assertEqual(conn.is_connected, True)

                # Test 2: Verify is_connected is False after close on previously opened connection
                conn.close()
                self.assertEqual(conn.is_connected, False)

            except Exception as e:
                self.fail(f'unexpected exception: {e}')

        # TODO: Is there a way to verify the options on a ZMQ socket????

    def test_receive(self):
        try:
            conn = GodotConnection(port=10001, type=GodotConnectionType.OBSERVATIONS)
            conn.open()

            message = conn.receive(max_tries=1)
            self.assertIsNotNone(message)
        except Exception as e:
            self.fail(f'unexpected exception: {e}')

    def test_send(self):
        try:
            conn = GodotConnection(port=10002, type=GodotConnectionType.ACTIONS)
            conn.open()

            message = {
                'header': {'agent_id': '1'},
                'data': {}
            }

            reply = conn.send(message={'agent_id': 1, 'action': 'up'})
            self.assertEqual(reply['status'], 'SUCCESS')

        except Exception as e:
            self.fail(f'unexpected exception: {e}')


if __name__ == '__main__':
    unittest.main()
