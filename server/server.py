import json
import logging
import os
import threading
import time
import uuid
from typing import Tuple, Union

import cv2
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiohttp import web
import asyncio

import fractions

from av.frame import Frame
from av.packet import Packet

from base import VoitsTTS, video_queue, audio_queue

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
sovits = VoitsTTS()
AUDIO_PTIME = 0.020
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self._queue = asyncio.Queue()
        self.timelist = []  # 记录最近包的时间戳
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                # self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('video start:', self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else:  # audio
            if hasattr(self, "_timestamp"):
                # self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                if wait > 0:
                    # print(wait)
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('audio start:', self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # print(self.kind)
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise Exception
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                print(f"------actual avg final fps:{self.framecount / self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0
        return frame

    def stop(self):
        super().stop()


class HumanPlayer:
    audio = PlayerStreamTrack(kind="audio")
    video = PlayerStreamTrack(kind="video")


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r", encoding='utf-8').read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r", encoding='utf-8').read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    daemon_thread = threading.Thread(target=run_coroutine_in_thread)
    daemon_thread.daemon = True  # 设置为守护线程，主线程退出时，守护线程也会退出
    daemon_thread.start()
    pc.addTrack(HumanPlayer.audio)
    pc.addTrack(HumanPlayer.video)

    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def human(request):
    text = request.query.get('text', '')  # 如果没有提供 'text' 参数，默认值为空字符串
    await interface(text)
    return web.Response(content_type="application/json", text=json.dumps({"status": "success", "text": text}))


async def interface(text):
    result_generator = sovits.gpt_sovits(
        text,
        'xiaofu.mp3', '你好，我是小福，我是一个人工助手', 'zh',
        'http://192.168.1.13:9881')
    ad_stream = sovits.stream_tts(result_generator)
    thread = threading.Thread(target=sovits.put_video, args=(ad_stream,))
    thread.start()  # 启动线程

    thread2 = threading.Thread(target=sovits.put_audio, args=(ad_stream,))
    thread2.start()  # 启动线程


async def listen_queue():
    img_index = 0
    step_stride = 0
    frame_directory = 'frames'
    while True:
        if video_queue.qsize() > 0 and audio_queue.qsize() > 1:
            # print(video_queue.qsize())
            await asyncio.sleep(0.01)
            frame = video_queue.get(block=False, timeout=0.01)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为RGB格式
            frame = np.array(frame)
            frame = VideoFrame.from_ndarray(frame, format="rgb24")
            await HumanPlayer.video._queue.put(frame)
            for di in range(2):
                audio_frame = audio_queue.get(block=False, timeout=0.01)
                audio_frame = (audio_frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=audio_frame.shape[0])
                new_frame.planes[0].update(audio_frame.tobytes())
                new_frame.sample_rate = 16000
                audio_frame = new_frame
                await HumanPlayer.audio._queue.put(audio_frame)
        else:
            # 加载当前帧
            await asyncio.sleep(0.01)
            vsize = HumanPlayer.video._queue.qsize()
            if vsize > 10:
                time.sleep(0.01)
                continue
            frame_files = sorted(os.listdir(frame_directory))  # 获取所有图片文件，按名称排序
            fsize = len(frame_files)
            frame_path = os.path.join(frame_directory, frame_files[img_index])
            frame = cv2.imread(frame_path)
            if img_index >= fsize - 1:
                step_stride = -1
            if img_index < 1:
                step_stride = 1
            img_index += step_stride
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为RGB格式
            frame = np.array(frame)
            frame = VideoFrame.from_ndarray(frame, format="rgb24")
            await HumanPlayer.video._queue.put(frame)
            # 塞入静音音频
            for i in range(2):
                audio_frame = np.zeros(320, dtype=np.float32)
                audio_frame = audio_frame.astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=audio_frame.shape[0])
                new_frame.planes[0].update(audio_frame.tobytes())
                new_frame.sample_rate = 16000
                audio_frame = new_frame
                await HumanPlayer.audio._queue.put(audio_frame)


def run_coroutine_in_thread():
    asyncio.run(listen_queue())


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    app.router.add_get('/human', human)
    web.run_app(
        app, access_log=None, host="0.0.0.0", port=8080)
