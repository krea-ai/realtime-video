import socketio
import json
import time
import argparse

# Create a Socket.IO client
sio = socketio.Client()

# ---- Event handlers for visibility ----
@sio.event
def connect():
    print('[client] connected')


@sio.event
def disconnect():
    print('[client] disconnected')


@sio.on('status')
def on_status(data):
    print('[server status]', data)


@sio.on('generation_params_updated')
def on_generation_params_updated(data):
    print('[server generation_params_updated]', data)


@sio.on('error')
def on_error(data):
    print('[server error]', data)


# Connect to the server
sio.connect('http://localhost:8000')

# Brief delay to ensure connection is fully established
time.sleep(0.2)

# Payload to send
example_payload = {
    'prompt': 'A beautiful sunset over mountains',
    'prompt2': '',
    'do_prompt_expand': True,
    'num_blocks': 7,
    'interp_blocks': -1,
    'context_noise': 0.0,
    'kv_cache_scale': 1.0,
    'kv_cache_num_frames': 9,
    'keep_first_frame': True,
    'kv_scale_allow_first': True,
    'kv_cache_num_frames_sequence': [],
    'timestep_to_apply_kv_cache_scale': [0, 1],
    'dynamic_timestep_to_apply_kv_cache_scale': None,
    'seed': 42,
    'enable_torch_compile': True,
    'enable_fp8': False,
    'use_taehv': False,
    'fps': 16,
    'sinusoidal_kv_enabled': False,
    'do_kv_recomp': False,
    # 'enable_'
    # 'sinusoidal_min_value': ,
    # 'sinusoidal_max_value': 0.7,
    # 'sinusoidal_frequency': 1.0
}

settings_update = {
    'v2v_steps': 4,
    'do_prompt_expand': False,
    'kv_cache_num_frames': 6,
    'thresh_kv_scale': 1.0,
}

parser = argparse.ArgumentParser(description='Socket.IO client for video generation')
parser.add_argument('--type', choices=['update', 'start', 'stop'], required=True,
                   help='Type of action to perform')
args = parser.parse_args()
print('type =', args.type)

if args.type == 'start':
    sio.emit('/session/1', example_payload)
    # Give the server time to start generation and respond
    time.sleep(1.0)
elif args.type == 'update':
    sio.emit('update_generation_params', settings_update)
    # Wait briefly to allow delivery and server processing
    time.sleep(0.5)
elif args.type == 'stop':
    sio.emit('stop_generation')
    time.sleep(0.2)

# Disconnect after giving time for any async events
sio.disconnect()
