# Скачиваем библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import h2o
from h2o.automl import H2OAutoML
from flaml import AutoML


################################################# Читаем CSV файл в pandas DataFrame
df = pd.read_csv("2022.csv")
# Преобразуем названия стадий сна в числа
# Бодрствование - 0, Лёгкий сон - 1, Глубокий сон - 2, Пробуждение - 3
df['Стадия'] = df['Стадия'].map({'Бодрствование':0, 'Лёгкий сон':1, 'Глубокий сон':2, 'Пробуждение':3})
# Удаляем ненужные колонки (полное время, дата, время)
df = df.drop(['full_time', 'date', 'time'], axis=1).dropna()
# Делим данные: 80% на обучение, 20% на тестирование
train, test = train_test_split(df, test_size=0.2, random_state=42)


##################################################### Запускаем H2O
h2o.init(verbose=False)
# Конвертируем pandas DataFrame в H2O формат
train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)
# Преобразуем целевую переменную в фактор  для задачи классификации
train_h2o['Стадия'] = train_h2o['Стадия'].asfactor()

# Создаём AutoML модель: максимум 180 секунд, фиксируем случайность (seed=42)
aml = H2OAutoML(max_runtime_secs=180, seed=42)
# Запускаем обучение, указываем целевую переменную
aml.train(y='Стадия', training_frame=train_h2o)
# Делаем предсказания на тестовых данных
pred = aml.leader.predict(test_h2o)

# Извлекаем предсказания в нужном формате:
# as_data_frame() → pandas DataFrame
# values.flatten() → плоский массив
# [:len(test)] → обрезаем до размера тестовой выборки (на всякий случай)
h2o_acc = accuracy_score(test['Стадия'], pred['predict'].as_data_frame().values.flatten()[:len(test)])
print(f"H2O: {h2o_acc:.4f} - {aml.leader.model_id}")
h2o.cluster().shutdown()
# Отделяем признаки (X) от целевой переменной (y) для train
X_train, y_train = train.drop('Стадия', axis=1), train['Стадия']
X_test, y_test = test.dr




######################################################### Создаём AutoML модель
automl = AutoML()
# Запускаем обучение:
# - task='classification' - задача классификации
# - time_budget=180 - максимум 180 секунд
automl.fit(X_train, y_train, task='classification', time_budget=180)

# Считаем точность: предсказали на тесте, сравнили с реальными ответами
flaml_acc = accuracy_score(y_test, automl.predict(X_test))
print(f"FLAML: {flaml_acc:.4f} - {automl.best_estimator}")



#Задача: определять, спит человек или нет, и если спит - то насколько глубоко. И использовали  пульс и текущее время. Получили точность 90.6% с помощью FLAML.
