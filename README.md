###  Описание проекта

Учебный проект, реализующий оркестрацию с помощью ApachAirflow, при решении задачи прогнозирования категории стоимости автомобиля.


С помощью Airflow последовательно выполняются две функции:

1. Функция pipeline из modules/pipeline.py
Осуществляет:
- загрузку данных;
- data cleaning;
- feature engineering;
- обучение трех моделей LogisticRegression, RandomForestClassifier, SVC;
- выбор лучшей модели на основе метрики и сохранение в файл.

2. Функция predict из modules/predict.py
Осуществляет получение предсказаний и запись их в файл.


### Установка зависимостей

```bash
pip install -r requirements.txt
