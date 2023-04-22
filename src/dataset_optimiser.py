# coding: utf8
class DatasetOptimiser:
    """
    Читает список филлеров из файла филлеров. Заменяет пробелы во всех многословных конструкциях в предложении
     на указанный символ
    """

    def __init__(self, fillers_file_path, delimiter=""):
        self.delimiter = delimiter
        self.fillers_replace_dict = {}
        with open(fillers_file_path, encoding='utf-8', mode='r') as fillers_file:
            lines = fillers_file.read().splitlines()
            lines = [line.split(' ') for line in lines if len(line.split(' ')) > 1]

            for line in lines:
                self.fillers_replace_dict[" ".join(line)] = delimiter.join(line)

            print(self.fillers_replace_dict)

    def process_sentence_str(self, sentence: str):
        result = sentence
        for old_word, new_word in self.fillers_replace_dict.items():
            result = result.replace(old_word, new_word)
        return result


if __name__ == "__main__":
    no_space_optimiser = DatasetOptimiser("../dataset/fillers.txt", "")
    dash_delimiter_optimiser = DatasetOptimiser("../dataset/fillers.txt", "-")
    underscore_delimiter_optimiser = DatasetOptimiser("../dataset/fillers.txt", "_")

    with open("../dataset/pos/text1.txt", encoding='utf-8', mode='r') as dataset_file_pos:
        lines = dataset_file_pos.readlines()
        with open("../dataset/pos/text1_no_space.txt", encoding='utf-8', mode='w') as output_no_space:
            for line in lines:
                output_no_space.write(no_space_optimiser.process_sentence_str(line))

        with open("../dataset/pos/text1_dash_delimiter.txt", encoding='utf-8', mode='w') as output_dash_delimiter:
            for line in lines:
                output_dash_delimiter.write(dash_delimiter_optimiser.process_sentence_str(line))

        with open("../dataset/pos/text1_underscore_delimiter.txt", encoding='utf-8',
                  mode='w') as output_underscore_delimiter:
            for line in lines:
                output_underscore_delimiter.write(underscore_delimiter_optimiser.process_sentence_str(line))

    with open("../dataset/neg/text1.txt", encoding='utf-8', mode='r') as dataset_file_neg:
        lines = dataset_file_neg.readlines()
        with open("../dataset/neg/text1_no_space.txt", encoding='utf-8', mode='w') as output_no_space:
            for line in lines:
                output_no_space.write(no_space_optimiser.process_sentence_str(line))

        with open("../dataset/neg/text1_dash_delimiter.txt", encoding='utf-8', mode='w') as output_dash_delimiter:
            for line in lines:
                output_dash_delimiter.write(dash_delimiter_optimiser.process_sentence_str(line))

        with open("../dataset/neg/text1_underscore_delimiter.txt", encoding='utf-8',
                  mode='w') as output_underscore_delimiter:
            for line in lines:
                output_underscore_delimiter.write(underscore_delimiter_optimiser.process_sentence_str(line))

    with open("../dataset/fillers.txt", encoding='utf-8', mode='r') as fillers_file:
        lines = fillers_file.readlines()
        with open("../dataset/fillers_no_space.txt", encoding='utf-8', mode='w') as output_no_space:
            for line in lines:
                output_no_space.write(no_space_optimiser.process_sentence_str(line))

        with open("../dataset/fillers_dash_delimiter.txt", encoding='utf-8', mode='w') as output_dash_delimiter:
            for line in lines:
                output_dash_delimiter.write(dash_delimiter_optimiser.process_sentence_str(line))

        with open("../dataset/fillers_underscore_delimiter.txt", encoding='utf-8',
                  mode='w') as output_underscore_delimiter:
            for line in lines:
                output_underscore_delimiter.write(underscore_delimiter_optimiser.process_sentence_str(line))
