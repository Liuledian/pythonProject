import pyaudio
import wave
import os
import requests
import uuid
import math

# record params
audio_dir = os.getenv("HOME") + "/test/audio-data"
chunk = 1024
channels = 1
fs = 16000
seconds = 3
# upload params
slice_size = 4 * 2**20
productId = "278589295"
apikey = "cc7d7c3a35654ed6bc7df213c78f9522"
create_audio_url = "https://lasr.duiopen.com/lasr-file-api/v2/audio"
upload_audio_url = "https://lasr.duiopen.com/lasr-file-api/v2/audio/{audio_id}/slice/{slice_index}"


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


def create_audio_on_server(slice_num):
    sid = str(uuid.uuid4())
    params = {'productId': productId,
              'apikey': apikey}
    headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
               'x-sessionId': sid}
    payload = {'audio_type': 'wav',
               'slice_num': slice_num}
    r = requests.post(create_audio_url, params=params, headers=headers, data=payload)
    print(r.json())
    audio_id = ""
    return audio_id, sid


def upload_audio(filename):
    file_size = os.path.getsize(filename)
    slice_num = math.ceil(file_size/slice_size)
    audio_id, sid = create_audio_on_server(slice_num)
    params = {'productId': productId,
              'apikey': apikey}
    headers = {'Content-Type': 'multipart/form-data',
               'x-sessionId': sid}
    with open(filename, 'rb') as audio_file:
        slice_index = 0
        while file_size > 0:
            read_size = min(file_size, slice_size)
            data = audio_file.read(read_size)
            r = requests.post(upload_audio_url.format(audio_id,slice_index),
                              params=params, headers=headers, files={'slice': data})
            print(r.text)
            file_size -= read_size
            slice_index += 1


if __name__ == '__main__':
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    files = os.listdir(audio_dir)
    index = len(files)
    record_audio_save(filename=audio_dir+"/{}.wav".format(index))
