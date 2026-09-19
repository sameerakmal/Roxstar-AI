import io
import logging
import asyncio
import av
import edge_tts
from livekit import rtc
from livekit.agents import tts, DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("edge-tts-wrapper")

class EdgeTTS(tts.TTS):
    def __init__(self, voice: str = "hi-IN-MadhurNeural", sample_rate: int = 24000, num_channels: int = 1):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self.voice = voice

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "EdgeTTSChunkedStream":
        return EdgeTTSChunkedStream(tts=self, input_text=text, conn_options=conn_options)

class EdgeTTSChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: tts.TTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._voice = getattr(tts, "voice", "hi-IN-MadhurNeural")
        self._text = input_text

    async def _run(self):
        try:
            communicate = edge_tts.Communicate(self._text, self._voice)
            mp3_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes.extend(chunk["data"])

            if not mp3_bytes:
                return

            container = av.open(io.BytesIO(mp3_bytes))
            resampler = av.AudioResampler(format="s16", layout="mono", rate=self._tts.sample_rate)

            for frame in container.decode(audio=0):
                for r_frame in resampler.resample(frame):
                    raw_pcm = r_frame.to_ndarray().tobytes()
                    rtc_frame = rtc.AudioFrame(
                        data=raw_pcm,
                        sample_rate=self._tts.sample_rate,
                        num_channels=self._tts.num_channels,
                        samples_per_channel=r_frame.samples,
                    )
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id="edge-tts",
                            segment_id="default",
                            frame=rtc_frame,
                        )
                    )

            for r_frame in resampler.resample(None):
                raw_pcm = r_frame.to_ndarray().tobytes()
                rtc_frame = rtc.AudioFrame(
                    data=raw_pcm,
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    samples_per_channel=r_frame.samples,
                )
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id="edge-tts",
                        segment_id="default",
                        frame=rtc_frame,
                    )
                )
        except Exception as e:
            logger.error(f"EdgeTTS synthesis failed: {e}")
