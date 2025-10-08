#!/usr/bin/env python3
"""
Simple WebSocket test client for the self-forcing video generation server.
Connects to the server and sends the same image every few seconds.
"""

import asyncio
import base64
import json 
import msgpack
import websockets

# Fill in the path to your test image
SEND_IMAGE = "/ceph/images/pd12m/a7c13a0f-ec83-5079-b098-e4ec175b165c.jpeg"
IMAGE_BYTES = open(SEND_IMAGE, "rb").read()
SAVE_IMAGES = False

async def main():
    uri = "ws://localhost:8000/session/test123"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server")
            
            # Send initial parameters
            params = {
                "prompt": "A cat riding a skateboard",
                "seed": 5366,
                "num_blocks": 1000,
            }
            # await websocket.send(msgpack.packb(params))
            print("Sent parameters")
            
            # Start listening for frames in background
            async def listen_for_frames():
                frame_count = 0
                async for message in websocket:
                    if isinstance(message, bytes):
                        frame_count += 1
                        print(f"Received frame {frame_count}")


                        if SAVE_IMAGES:
                            # Save frame to file
                            with open(f"client_outpus/frame_{frame_count:04d}.jpg", "wb") as f:
                                f.write(message)
            
            # Start frame listener
            listener_task = asyncio.create_task(listen_for_frames())
            
            # Send images periodically
            if SEND_IMAGE:
                while True:
                    await asyncio.sleep(3)  # Wait 3 seconds
                    # Send image frame as binary msgpack
                    frame_message = {"image": IMAGE_BYTES}
                    print("sending image bytes")
                    packed_data = msgpack.packb(frame_message)
                    await websocket.send(packed_data)
                    print("Sent image frame")
            else:
                print("SEND_IMAGE is empty - not sending any images")
                # Just wait and receive frames
                await listener_task
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())