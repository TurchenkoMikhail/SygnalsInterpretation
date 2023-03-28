import whisper_timestamped as whisper
import json


TARGET_WORD = "методов"


def get_segments_by_word(audio_file_path: str):
    start_time = []
    end_time = []
    audio = whisper.load_audio(audio_file_path)
    model = whisper.load_model("small", device="cpu")
    result = whisper.transcribe(model, audio, language="ru")
    #data = json.dumps(result, indent=2, ensure_ascii=False)
    print(result['text'])

    # Нахождение отрезков, содержащих TARGET_WORD
    for seg in result["segments"]:
        for w in seg["words"]:
            if TARGET_WORD in w["text"]:
                start_time.append(w["start"])
                end_time.append(w["end"])

    return start_time, end_time
