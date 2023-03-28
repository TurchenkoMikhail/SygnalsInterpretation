from pydub import AudioSegment


# вырезка интервалов из аудио
def cut(audio_file_path: str, start_time: [], end_time: []):
    audio = AudioSegment.from_mp3(audio_file_path)

    cur_start = 0
    cur_end = 0
    extracted_audio = AudioSegment.empty()
    for start, end in zip(start_time, end_time):
        extracted_audio += audio[cur_start * 1000: start * 1000]
        cur_start = end

    extracted_audio += audio[cur_start * 1000:]
    return extracted_audio