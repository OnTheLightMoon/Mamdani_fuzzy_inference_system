# -*- coding: cp1251 -*-

import sys
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt


# определяем переменные входных данных
food = np.arange(1, 11, 1)
service = np.arange(1, 11, 1)
tip = np.arange(1, 26, 1)


# определяем функции принадлежности
food_rancid = fuzz.zmf(food, 3, 8)
food_delicious = fuzz.smf(food, 3, 8)

service_poor = fuzz.zmf(service, 3, 5)
service_good = fuzz.trapmf(service, [3, 5, 6, 8])
service_excellent = fuzz.smf(service, 6, 8)

tip_cheap = fuzz.zmf(tip, 4, 13)
tip_average = fuzz.trapmf(tip, [4, 13, 14, 21])
tip_generous = fuzz.smf(tip, 14, 21)


# пользователь вводит входные данные
food_value = float(input("На сколько баллов от 1 до 10 вы оцениваете еду? "))
service_value = float(input("На сколько баллов от 1 до 10 вы оцениваете сервис? "))


# нахождение степеней принадлежностей для введенных значений
food_rancid_level = fuzz.interp_membership(food, food_rancid, food_value)
food_delicious_level = fuzz.interp_membership(food, food_delicious, food_value)
service_poor_level = fuzz.interp_membership(service, service_poor, service_value)
service_good_level = fuzz.interp_membership(service, service_good, service_value)
service_excellent_level = fuzz.interp_membership(service, service_excellent, service_value)


# правила нечеткой логики (возвращают минимум)
rule1 = np.fmin(food_delicious_level, service_poor_level) #низкие чаевые
rule2 = np.fmin(food_delicious_level, service_good_level) #средние чаевые
rule3 = np.fmin(food_delicious_level, service_excellent_level) #высокие чаевые
rule4 = np.fmin(food_rancid_level, service_poor_level) #низкие чаевые
rule5 = np.fmin(food_rancid_level, service_good_level) #низкие чаевые
rule6 = np.fmin(food_rancid_level, service_excellent_level) #средние чаевые


# степени принадлежностей
tip_cheap_level = np.fmin(tip_cheap, np.fmax(rule1, np.fmax(rule4, rule5)))
tip_average_level = np.fmin(tip_average, np.fmax(rule2, rule6))
tip_generous_level = np.fmin(tip_generous, rule3)


# агрегирование выводов всех правил
tip_aggregated0 = fuzz.fuzzy_or(tip, tip_cheap_level, tip, tip_average_level)
tip_aggregated = fuzz.fuzzy_or(tip, tip_aggregated0[1], tip, tip_generous_level)


# расчет чаевых на основе агрегированного вывода
tip_value = fuzz.defuzz(tip, tip_aggregated[1], 'centroid')
tip_activation = fuzz.interp_membership(tip, tip_aggregated[1], tip_value)


# вывод результ
print("Чаевые:", round(tip_value,1))

#задание параметров для виуализации графиков входных и выходной перменных
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 7))

ax0.plot(food, food_rancid, 'c', linewidth=1.5, label='невкусно')
ax0.plot(food, food_delicious, 'b', linewidth=1.5, label='вкусно')
ax0.set_title('Качество еды')
ax0.legend()

ax1.plot(service, service_poor, 'c', linewidth=1.5, label='плохое')
ax1.plot(service, service_good, 'g', linewidth=1.5, label='среднее')
ax1.plot(service, service_excellent, 'b', linewidth=1.5, label='отличное')
ax1.set_title('Качество обсуживания')
ax1.legend()

ax2.plot(tip, tip_cheap, 'c', linewidth=1.5, label='низкое')
ax2.plot(tip, tip_average, 'g', linewidth=1.5, label='среднее')
ax2.plot(tip, tip_generous, 'b', linewidth=1.5, label='высокое')
ax2.set_title('Количество чаевых')
ax2.legend()

#отключение осей
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

#коррекция вывода графиков
plt.tight_layout()

tip0 = np.zeros_like(tip)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(tip, tip0, tip_cheap_level, facecolor='c', alpha=0.9)
ax0.plot(tip, tip_cheap, 'c', linewidth=0.7, linestyle='--', )
ax0.fill_between(tip, tip0, tip_average_level, facecolor='g', alpha=0.9)
ax0.plot(tip, tip_average, 'g', linewidth=0.7, linestyle='--')
ax0.fill_between(tip, tip0, tip_generous_level, facecolor='b', alpha=0.9)
ax0.plot(tip, tip_generous, 'b', linewidth=0.7, linestyle='--')
ax0.set_title('Фаззификация')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(tip, tip_cheap_level, 'c', linewidth=0.7, linestyle='--', )
ax0.plot(tip, tip_average_level, color='g', linewidth=0.5, linestyle='--')
ax0.plot(tip, tip_generous_level, 'b', linewidth=0.7, linestyle='--')
ax0.fill_between(tip, tip0, tip_aggregated[1], facecolor='indigo', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Дефаззификация')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
