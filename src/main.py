from SygnalsInterpretation.src.audio_handler import audio_handler, transcribe

"""Машинное обучение — класс методов искусственного интеллекта, 
характерной чертой которых является непрямое решение задачи, 
а обучение — за счет применения решений множество сходных задач. 
Для построения таких методов используется средство математической статистики, численных методов, 
математического анализа, методов оптимизации, различные техники работы с данными в цифровой форме."""
"""                     ||                      """
"""                     \/                      """
audio_file_path = "dataset/audio_test.mp3"


if __name__ == "__main__":
    # Распознание
    start, end = transcribe.get_segments_by_word(audio_file_path)

    # Вырезание
    ex_audio = audio_handler.cut(audio_file_path, start, end)
    ex_audio.export(f"ans.mp3", format="mp3")




