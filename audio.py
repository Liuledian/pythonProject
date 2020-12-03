import pyaudio
import wave
import os
import requests
import uuid

audio_dir = os.getenv("HOME") + "/test/audio-data"
chunk = 1024
channels = 1
fs = 16000
seconds = 3


def record_audio_save(filename):
    sample_format = pyaudio.paInt16  # 16 bits per sample
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        print("chunk{}".format(i))
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def create_audio_on_server():
    url = "https://lasr.duiopen.com/lasr-file-api/v2/audio"
    sid = str(uuid.uuid4())
    headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
              'x-sessionId': sid}
    payload = {'audio_type': 'wav', 'slice_num': 1}
    r = requests.post(url, headers=headers, data=payload)
    print(r.json())


if __name__ == '__main__':
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    files = os.listdir(audio_dir)
    index = len(files)
    record_audio_save(filename=audio_dir+"/{}.wav".format(index))
