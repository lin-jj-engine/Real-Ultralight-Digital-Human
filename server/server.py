import json
import logging
import multiprocessing
import os
import threading
import time
import uuid
from typing import Tuple

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack
from aiohttp import web
import asyncio

from av import AudioFrame
import fractions

from base import VoitsTTS, audio_queue, video_queue

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
sovits = VoitsTTS()
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 21  # fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class FrameStreamTrack(VideoStreamTrack):
    kind = 'video'  # 确保轨道类型是视频

    def __init__(self):
        super().__init__()
        self.index = 0
        self.start_frames = 30
        self.step_stride = 0
        self.max_frame = 48

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self):
        print("1")
        try:
            print(video_queue.qsize())
            if audio_queue.qsize() < 1 or video_queue.qsize() < self.start_frames:
                raise ZeroDivisionError
            if video_queue.qsize() >= self.start_frames:
                self.start_frames = 0
            if video_queue.qsize() == 0:
                self.start_frames = 30
            frame = video_queue.get(block=False, timeout=0.01)
        except Exception:
            frame_directory = 'frames'
            # 加载当前帧
            frame_files = sorted(os.listdir(frame_directory))  # 获取所有图片文件，按名称排序
            fsize = len(frame_files)
            frame_path = os.path.join(frame_directory, frame_files[self.index])
            frame = cv2.imread(frame_path)
            if self.index >= fsize - 1:
                self.step_stride = -1
            if self.index < 1:
                self.step_stride = 1
            self.index += self.step_stride
            if self.index > fsize - 1:
                self.index = fsize - 1

        print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为RGB格式
        frame = np.array(frame)
        frame = VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        print(frame.pts, frame.time_base)
        return frame


class AudioFileTrack(AudioStreamTrack):
    """
    从 WAV 文件中读取音频数据的自定义音频轨道。
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.sovits = VoitsTTS()
        self.sample_rate = 16000
        self._timestamp = 0  # 初始化时间戳

    async def recv(self):
        """
        从 WAV 文件读取音频帧，并返回音频帧。
        """

        try:
            frame = audio_queue.get(block=False, timeout=0.01)
        except Exception:
            frame = np.zeros(320, dtype=np.float32)
        frame = (frame * 32767).astype(np.int16)
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = 16000
        frame = new_frame
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        # print(frame.pts / self.sample_rate)
        self._timestamp += 320  # 增加时间戳
        time.sleep(0.01)
        return frame


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

    audio_track = AudioFileTrack()  # 设置视频路径
    pc.addTrack(audio_track)
    pc.addTrack(FrameStreamTrack())
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
    thread1 = threading.Thread(target=sovits.put_video, args=(ad_stream,))
    thread1.start()  # 启动线程
    thread1 = threading.Thread(target=sovits.put_audio, args=(ad_stream,))
    thread1.start()  # 启动线程


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
