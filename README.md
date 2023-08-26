# Mamdani_fuzzy_inference_system
## Цель работы
Решить практическую задачу с применением нечёткого вывода Мамдани.

## Задача
Вы после учебного/рабочего дня решили зайти пообедать в кафе. Вы выбрали первое кафе, которое Вам понравилось, но не обладаете никакой информацией, насколько вкусно там готовят и хороший ли там сервис. То есть выбор данного кафе связан с неопределённостью.
Степень своей удовлетворённости от кафе Вы будете выражать в размере чаевых, которые колеблются в интервале от 0 до 25% от счёта заказа. Будем полагать, что щедрость чаевых будет зависеть от двух факторов, которые обозначим, как:
1. Качество еды (то есть насколько она была хорошо приготовленной, свежей и прочее), которое будет оцениваться как:
-	вкусно;
-	невкусно.
2. Качество обслуживания (то есть насколько быстро был оформлен заказ, был ли чистым стол и прочее), оцениваемое по шкале:
-	отличное;
-	среднее;
-	плохое.
  
Пусть по итогам посещения кафе Вы пришли к выводу, что еда была достаточно хорошей, а вот сервис был на низком уровне, то есть плохим.
Возникает вопрос – сколько же дать чаевых официанту в зависимости от выбранных параметров?
## Ход работы
В результате опроса экспертной группы были получены данные, пока-занные на рисунках 1 – 3.
![image](https://github.com/OnTheLightMoon/Mamdani_fuzzy_inference_system/assets/143195378/e3327708-afe9-4b38-a090-d75e7e6232e8)

*Рисунок 1 – Оценка качества еды*

 ![image](https://github.com/OnTheLightMoon/Mamdani_fuzzy_inference_system/assets/143195378/0d72455d-0c70-4c66-ade8-d442a10e5b2c)

*Рисунок 2 – Оценка качества обслуживания*

 ![image](https://github.com/OnTheLightMoon/Mamdani_fuzzy_inference_system/assets/143195378/7bb3fb2e-2f06-4d58-b1e0-4cacc4ee32fb)

*Рисунок 3 – Оценка чаевых*

Первым делом импортируем необходимые нам библиотеки. Для реали-зации нечеткой логики будем использовать библиотеку skfuzzy 0.2 (Листинг 1).

*Листинг 1 – Код для импорта необходимых библиотек*

    import sys
    import numpy as np
    import skfuzzy as fuzz
    from matplotlib import pyplot as plt

Далее необходимо задать возможные значения входных переменных, то есть базовые множества (Листинг 2).

*Листинг 2 – Код для задания базовых множеств*

    #определяем переменные входных данных
    food = np.arange(1, 11, 1)
    service = np.arange(1, 11, 1)
    tip = np.arange(1, 26, 1)
    
Определяем функции принадлежности на основе результатов опроса экспертной группы (Листинг 3).

*Листинг 3 – Код для задания функций принадлежности*

    #определяем функции принадлежности
    food_rancid = fuzz.zmf(food, 3, 8)
    food_delicious = fuzz.smf(food, 3, 8)

    service_poor = fuzz.zmf(service, 3, 5)
    service_good = fuzz.trapmf(service, [3, 5, 6, 8])
    service_excellent = fuzz.smf(service, 6, 8)

    tip_cheap = fuzz.zmf(tip, 4, 13)
    tip_average = fuzz.trapmf(tip, [4, 13, 14, 21])
    tip_generous = fuzz.smf(tip, 14, 21)
    
На графиках функции принадлежности будут выглядеть следующим об-разом (Рисунки 4-6):

 ![image](https://github.com/OnTheLightMoon/Mamdani_fuzzy_inference_system/assets/143195378/f5ec37e2-d728-45c7-a960-dc3edd813039)

*Рисунок 4 – Функции принадлежности для еды*

 ![image](https://github.com/OnTheLightMoon/Mamdani_fuzzy_inference_system/assets/143195378/1dcc34c6-da68-42a1-9540-cd7178219191)

*Рисунок 5 – Функции принадлежности для сервиса*

 ![image](https://github.com/OnTheLightMoon/Mamdani_fuzzy_inference_system/assets/143195378/078dad00-ba0e-46ab-8b57-2cff5e379061)

*Рисунок 6 – Функции принадлежности для чаевых*

Напишем код, с помощью которого у пользователя будет возможность вводить балл, на который оценивает качество еды и сервиса (Листинг 4).

*Листинг 4 – Код для задания функций принадлежности*

    #пользователь вводит входные данные
    food_value = float(input("Enter food value (1-10): "))
    service_value = float(input("Enter service value (1-10): "))

Далее находим степени принадлежности для введенных пользователем значений для каждого терм-множества (Листинг 5).

*Листинг 5 – Код для задания функций принадлежности*

    #нахождение степеней принадлежностей для введенных значений
    food_rancid_level=fuzz.interp_membership(food,food_rancid,food_value)
    food_delicious_level=fuzz.interp_membership(food,food_delicious,food_value) 
    service_poor_level=fuzz.interp_membership(service,service_poor,service_value)
    service_good_level=fuzz.interp_membership(service,service_good,service_value)
    service_excellent_level=fuzz.interp_membership(service,service_excellent,service_value)

Теперь необходимо прописать правила, на основе которых будут рас-считываться чаевые. Всего будет 6 правил:

- не вкусная еда + плохое обслуживание = низкие чаевые
-	не вкусная еда + среднее обслуживание = средние чаевые
-	не вкусная еда + высокое обслуживание = высокие чаевые
-	вкусная еда + плохое обслуживание = низкие чаевые
-	вкусная еда + среднее обслуживание = низкие чаевые
-	вкусная еда + высокое обслуживание = средние чаевые
  
Находим степени истинности для каждого из заданных правил, после чего отсекаем функции принадлежности заключений заданных правил на уровнях степеней истинностей (Листинг 6).

*Листинг 6 – Код для задания базы правил*

    #правила нечеткой логики (возвращают минимум)
    rule1 = np.fmin(food_delicious_level, service_poor_level) #низкие чаевые
    rule2 = np.fmin(food_delicious_level, service_good_level) #средние чаевые
    rule3 = np.fmin(food_delicious_level, ser-vice_excellent_level) #высокие чаевые
    rule4 = np.fmin(food_rancid_level, service_poor_level) #низкие чаевые
    rule5 = np.fmin(food_rancid_level, service_good_level) #низкие чаевые
    rule6 = np.fmin(food_rancid_level, service_excellent_level) #средние чаевые
    #степени принадлежностей
    tip_cheap_level = np.fmin(tip_cheap, np.fmax(rule1, np.fmax(rule4, rule5)))
    tip_average_level = np.fmin(tip_average, np.fmax(rule2, rule6))
    tip_generous_level = np.fmin(tip_generous, rule3)
    
Далее производим агрегирование выводов всех правил посредством объединения (Листинг 7).

*Листинг 7 – Код для агрегирования выводов*

    #агрегирование выводов всех правил
    tip_aggregated0 = fuzz.fuzzy_or(tip, tip_cheap_level, tip, tip_average_level)
    tip_aggregated = fuzz.fuzzy_or(tip, tip_aggregated0[1], tip, tip_generous_level)

На последнем этапе производим дефаззификацию относительно центра области (Листинг 8).

*Листинг 8 – Код для дефаззификации*

    #расчет чаевых на основе агрегированного вывода
    tip_value = fuzz.defuzz(tip, tip_aggregated[1], 'centroid')
    
Выводим значение чаевых пользователю в консоль, предварительно округлив его до десятых (Листинг 9).

*Листинг 9 – Вывод результата*

    #вывод результ
    print("Tip value:", round(tip_value,1))
