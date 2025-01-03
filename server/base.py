import time
from io import BytesIO
from typing import Iterator
import resampy
import numpy as np
import requests
from queue import Queue
import soundfile as sf

from utils import wav2vec2_processor, hubert_model, infer, get_hubert_from_16k_speech, make_even_first_dim

audio_queue = Queue()
video_queue = Queue()


class VoitsTTS():
    sample_rate = 16000
    chunk = sample_rate // 50

    def gpt_sovits(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': language,
            'ref_audio_path': reffile,
            'prompt_text': reftext,
            'prompt_lang': language,
            'media_type': 'ogg',
            'streaming_mode': True
        }

        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            print(f"gpt_sovits Time to make POST: {end - start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True

            for chunk in res.iter_content(chunk_size=None):  # 12800 1280 32K*20ms*2
                print('chunk len:', len(chunk))
                if first:
                    end = time.perf_counter()
                    print(f"gpt_sovits Time to first chunk: {end - start}s")
                    first = False
                if chunk:
                    yield chunk
            # print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            print(e)

    def create_bytes_stream(self, byte_stream):
        # byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream)  # [T*sample_rate,] float64
        # sf.write('res.ogg', stream, sample_rate, format='OGG', subtype='VORBIS')

        print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
        return stream

    def stream_tts(self, audio_stream):
        streams = []
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                byte_stream = BytesIO(chunk)
                stream = self.create_bytes_stream(byte_stream)
                streams.append(stream)
        return streams

    def put_audio(self, streams):
        time.sleep(3)  # 显卡不行，模型跑得快的可以删掉
        for stream in streams:
            streamlen = stream.shape[0]
            idx = 0
            while streamlen >= self.chunk:
                frame = stream[idx:idx + self.chunk]
                audio_queue.put(frame)
                streamlen -= self.chunk
                idx += self.chunk

    def put_video(self, streams):
        for stream in streams:
            streamlen = stream.shape[0]
            if streamlen > 640:
                hubert_hidden = get_hubert_from_16k_speech(hubert_model, wav2vec2_processor, stream)
                hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
                infer(hubert_hidden.detach().numpy(), queue=video_queue)
