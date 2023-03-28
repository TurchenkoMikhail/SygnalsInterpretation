## Что используем?
Используем библиотеку [Whisper](https://github.com/openai/whisper) от OpenAI

Однако она не разбивает таймкоды по словам, а только лишь по фразам, поэтому используем [надстройку Whisper](https://github.com/linto-ai/whisper-timestamped)

С помощью `whisper-timestamped` получаем примерные таймкоды слов. 

## Как запустить?
Достаточно скачать библиотеку `whisper-timestamped` по инструкции, указанной на Github в источнике \
```pip3 install git+https://github.com/linto-ai/whisper-timestamped```

Пример [аудио](src/audio_test.mp3) - вырезается слово "методов" из аудиодорожки. 
В функции main в комментарии указан текст аудио.