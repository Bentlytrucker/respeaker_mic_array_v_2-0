import pyaudio
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1 # change base on firmwares, default_firmware.bin as 1 or i6_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 1  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

def record_audio():
    """마이크 배열로부터 오디오를 녹음합니다."""
    p = pyaudio.PyAudio()

    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)

    print("* recording")

    frames = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # WAV 파일로 저장
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return WAVE_OUTPUT_FILENAME

def create_mel_spectrogram(audio_file, n_mels=128, hop_length=512, n_fft=2048):
    """
    오디오 파일을 mel spectrogram으로 변환합니다.
    
    Args:
        audio_file (str): 오디오 파일 경로
        n_mels (int): mel 필터뱅크의 개수
        hop_length (int): 프레임 간 이동 샘플 수
        n_fft (int): FFT 윈도우 크기
    
    Returns:
        tuple: (mel_spectrogram, sample_rate)
    """
    # 오디오 로드
    y, sr = librosa.load(audio_file, sr=RESPEAKER_RATE)
    
    # Mel spectrogram 계산
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels, 
        hop_length=hop_length, 
        n_fft=n_fft
    )
    
    # 데시벨 스케일로 변환
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db, sr

def visualize_mel_spectrogram(mel_spectrogram_db, sr, title="Mel Spectrogram"):
    """
    Mel spectrogram을 시각화합니다.
    
    Args:
        mel_spectrogram_db (np.ndarray): 데시벨 스케일의 mel spectrogram
        sr (int): 샘플링 레이트
        title (str): 그래프 제목
    """
    plt.figure(figsize=(12, 8))
    
    # Mel spectrogram 시각화
    librosa.display.specshow(
        mel_spectrogram_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

def save_mel_spectrogram(mel_spectrogram_db, filename="mel_spectrogram.png"):
    """
    Mel spectrogram을 이미지 파일로 저장합니다.
    
    Args:
        mel_spectrogram_db (np.ndarray): 데시벨 스케일의 mel spectrogram
        filename (str): 저장할 파일명
    """
    plt.figure(figsize=(12, 8))
    
    librosa.display.specshow(
        mel_spectrogram_db, 
        sr=RESPEAKER_RATE, 
        x_axis='time', 
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mel spectrogram saved as {filename}")

if __name__ == "__main__":
    # 오디오 녹음
    audio_file = record_audio()
    
    # Mel spectrogram 생성
    print("Creating mel spectrogram...")
    mel_spec_db, sr = create_mel_spectrogram(audio_file)
    
    # Mel spectrogram 시각화
    print("Visualizing mel spectrogram...")
    visualize_mel_spectrogram(mel_spec_db, sr)
    
    # Mel spectrogram 저장
    print("Saving mel spectrogram...")
    save_mel_spectrogram(mel_spec_db)
    
    print("Mel spectrogram processing completed!")
