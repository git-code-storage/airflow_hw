import os
import dill
import json
import pandas as pd
import datetime

def predict():

    project_path = os.path.expanduser('/home/vadim/Airflow')

    # Чтение тестовых данных
    list_of_test_dict = []
    test_path = project_path + '/data/test'
    list_of_test_files = os.listdir(test_path)
    for test_file in list_of_test_files:
        with open(os.path.expanduser(test_path + f'/{test_file}'), 'r') as file:
            test_dict = json.load(file)
        list_of_test_dict.append(test_dict)

    # Чтение крайнего файла модели
    models_path = project_path + '/data/models'
    last_model = models_path + f'/{sorted(os.listdir(models_path))[-1]}'
    with open(os.path.expanduser(last_model), 'rb') as file:
        model = dill.load(file)

    # Получение предсказаний
    list_of_predicts = []
    for test_dict in list_of_test_dict:
        one_predict_dict = dict()
        df = pd.DataFrame.from_dict([test_dict])
        predict = model.predict(df)
        one_predict_dict['predict'] = f'{predict[0]}'
        time = datetime.datetime.now()
        one_predict_dict['time'] = f'{time.year}-{time.month}-{time.day} {time.hour}:{time.minute}:{time.second}'
        list_of_predicts.append(one_predict_dict)
    df_new_pred = pd.DataFrame(list_of_predicts)

    # Запись предсказаний в файл
    df_read = pd.read_csv(project_path + '/data/predictions/predict.csv')
    df_concat = pd.concat([df_read, df_new_pred], axis=0, ignore_index=True)
    df_result = df_concat[['predict', 'time']]
    df_result.to_csv(project_path + '/data/predictions/predict.csv')


if __name__ == '__main__':
    predict()
