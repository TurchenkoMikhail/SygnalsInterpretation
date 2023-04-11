from pydub import AudioSegment


class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


# вырезка интервалов из аудио
def cut(audio_file_path: str, start_time_all: [], end_time_all: []):
    audio = AudioSegment.from_mp3(audio_file_path)
    extracted_audio = AudioSegment.empty()
    time_segs = list()

    for start_time, end_time in zip(start_time_all, end_time_all):
        for start, end in zip(start_time, end_time):
            time_segs.append(Segment(start, end))

    time_segs = sorted(time_segs, key=lambda x: x.start)

    cur_start = 0
    for seg in time_segs:
        extracted_audio += audio[cur_start * 1000: seg.start * 1000]
        cur_start = seg.end

    extracted_audio += audio[cur_start * 1000:]

    return extracted_audio
