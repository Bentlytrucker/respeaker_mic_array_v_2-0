import pyaudio
import numpy as np
import time

def test_multi_channel():
    """6채널 펌웨어 테스트"""
    p = pyaudio.PyAudio()
    
    # 6채널 설정
    CHANNELS = 6
    RATE = 16000
    CHUNK = 1024
    
    # ReSpeaker 디바이스 찾기
    device_index = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'ReSpeaker 4 Mic Array' in dev['name'] and dev['maxInputChannels'] == CHANNELS:
            device_index = i
            break
    
    if device_index is None:
        print("6채널 ReSpeaker 디바이스를 찾을 수 없습니다.")
        return
    
    print(f"6채널 디바이스 발견: {device_index}")
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    
    print("6채널 오디오 스트림 시작...")
    print("5초간 각 채널의 오디오 레벨을 모니터링합니다.")
    print("채널 0: 처리된 오디오")
    print("채널 1-4: 각 마이크 원시 데이터")
    print("채널 5: 재생 채널")
    
    for i in range(5):
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # 각 채널별로 데이터 분리
        rms_values = []
        for ch in range(CHANNELS):
            channel_data = audio_data[ch::CHANNELS]
            rms = np.sqrt(np.mean(np.square(channel_data.astype('float32'))))
            rms_values.append(rms)
        
        print(f"초 {i+1}: {rms_values}")
        time.sleep(1)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("6채널 테스트 완료")

if __name__ == "__main__":
    test_multi_channel()
