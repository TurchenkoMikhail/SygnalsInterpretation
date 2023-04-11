import copy

import whisper_timestamped as whisper
from SygnalsInterpretation.src.model.model import fit, vocabulary
import json


TARGET_WORD = "методов"


def get_sentences(sentences):
    count = 0
    sentence_build = []
    cur_sentence = []
    for seg in sentences["segments"]:
        cur_sentence.append(seg["words"])
        if "." in seg["text"]:
            sentence_build.append(copy.deepcopy(cur_sentence))
            cur_sentence.clear()
            count += 1
    if len(cur_sentence) != 0:
        sentence_build.append(cur_sentence)
    return sentence_build


def cut_segments(sentence: list, indices):
    start_time = []
    end_time = []
    for part in sentence:
        for word in part:
            for index in indices:
                if vocabulary[index] in word["text"].lower():
                    start_time.append(word["start"])
                    end_time.append(word["end"])

    return start_time, end_time


def cut_fillers(audio_file_path: str, error=0.8):
    sentences = load(audio_file_path)
    sentences = get_sentences(sentences)
    start_all = []
    end_all = []

    clf, vectorizer = fit()
    for sentence in sentences:
        input = ' '.join(word['text'] for part in sentence for word in part)
        input = input.lower()
        print(input)
        X = vectorizer.transform([input])
        probability = clf.predict_proba(X)
        if probability.max() > error:
            start_time, end_time = cut_segments(sentence, X.indices)
            start_all.append(start_time)
            end_all.append(end_time)

    return start_all, end_all


def load(audio_file_path: str):
    audio = whisper.load_audio(audio_file_path)
    model = whisper.load_model("small", device="cpu")
    result = whisper.transcribe(model, audio, language="ru")
    return result


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
