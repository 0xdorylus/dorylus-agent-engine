
class AudioParams():
    # sample width
    WIDTH = 2
    # channels
    CHANNELS = 1
    # samples per second
    RATE = 16000
    # chunk duration(ms), support 10/20/30 ms
    CHUNK_DURATION_MS = 30
    # samples in chunk
    CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
    # chunks in voice starting detection window
    NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)
    # chunks in voice ending detection window
    NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
    # max recording seconds
    MAX_RECORDING_SECONDS = 60
    # move forward some chunks from voice found point
    START_OFFSET = max(20, NUM_WINDOW_CHUNKS)

    SIMPLE_VAD_AUDIO_TOTAL_MS = 3000
    SIMPLE_VAD_AUDIO_REAR_MS = 1500
    SIMPLE_VAD_AUDIO_CHUNK_SIZE = int(RATE * SIMPLE_VAD_AUDIO_TOTAL_MS / 1000)
    SIMPLE_VAD_THOLD = 0.3
    SIMPLE_VAD_FREQ_THOLD = 100.0

    STREAM_STEP_MS = 1500
    STREAM_LENGTH_MS = 9000
    STREAM_KEEP_MS = 300
    STREAM_SAMPLES_STEP = int(RATE * STREAM_STEP_MS / 1000)
    STREAM_SAMPLES_LEN = int(RATE * STREAM_LENGTH_MS / 1000)
    STREAM_SAMPLES_KEEP = int(RATE * STREAM_KEEP_MS / 1000)
    STREAM_STEPS_PER_LINE = int(STREAM_LENGTH_MS / STREAM_STEP_MS)
