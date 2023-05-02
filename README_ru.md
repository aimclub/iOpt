<p align="center">
  <img src="/docs/iOpt_logo.png" width="200" height="150"/>
</p>

[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-green)](LICENSE)
[![python: 3.9](https://img.shields.io/badge/python-3.9-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-390/)
[![python: 3.8](https://img.shields.io/badge/python-3.8-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-380/)
[![docs: ](https://readthedocs.org/projects/ebonite/badge/?style=flat-square)](https://iopt.readthedocs.io/ru/latest/)
[![build:](https://github.com/UNN-ITMM-Software/iOpt/actions/workflows/python-app.yml/badge.svg)](https://github.com/aimclub/iOpt/actions)
[![eng:](https://img.shields.io/badge/lang-en-red.svg)](README.md)



iOpt - фреймворк с открытым исходным кодом для автоматического выбора значений параметров как для математических моделей 
сложных промышленных процессов, так и для используемых в промышленности методов ИИ и МО. 
Фреймворк распространяется под лицензией 3-Clause BSD.

# **Основные возможности**
- Автоматический выбор значений параметров математических моделей и методов ИИ и МО, используемых в промышленности.
- Интеллектуальное управление процессом выбора оптимальных параметров для промышленных задач.
- Интеграция с внешними библиотеками или фреймворками искусственного интеллекта и машинного обучения, а также предметными моделями.
- Автоматизация предварительного анализа исследуемых моделей, например, выделение различных классов зависимостей модели от разных групп параметров.
- Визуализация процесса выбора оптимальных параметров.


# **Установка и настройка**

## В unix-подобных системах:

```
git clone https://github.com/UNN-ITMM-Software/iOpt
cd iOpt
pip install virtualenv
virtualenv ioptenv
source ioptenv/bin/activate
python setup.py install
```

## В ОС Windows:

```
git clone https://github.com/UNN-ITMM-Software/iOpt
cd iOpt
pip install virtualenv
virtualenv ioptenv
ioptenv\Scripts\activate.bat
python setup.py install
```


# **Начать работать**

Использование фреймворка iOpt для минимизации функции Растригина.

```python
from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Минимизация функции Растригина и визуализация
    """
    #Создание тестовой задачи
    problem = Rastrigin(2)
    #Установка параметров поиска оптимального решения
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)
    #Создание решателя
    solver = Solver(problem, parameters=params)
    #Вывод результатов на консоль в процессе решения
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)
    #3D визуализация по окончании решения
    spl = StaticNDPaintListener("rastrigin.png", "output", varsIndxs=[0,1], mode="surface", calc="interpolation")
    solver.AddListener(spl)
    #Запуск решения задачи
    sol = solver.Solve()
```

# **Примеры**

Продемонстрируем как использовать iOpt для настройки гиперпараметров модели машинного обучения.
В метода опорных векторов ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)) найдем оптимальные 
вещественные гиперпараметры (**C** - параметр регуляризации, **gamma** - коэффициент ядра) для решения задачи классификации рака молочной железы
 ([подробное описание данных](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))).


```python
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_breast_cancer

from iOpt.method.listener import StaticNDPaintListener, AnimationNDPaintListener, ConsoleFullOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import SVC_2d


def load_breast_cancer_data():
    dataset = load_breast_cancer()
    x_raw, y_raw = dataset['data'], dataset['target']
    inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
    return inputs, outputs


if __name__ == "__main__":
    x, y = load_breast_cancer_data()
    regularization_value_bound = {'low': 1, 'up': 6}
    kernel_coefficient_bound = {'low': -7, 'up': -3}

    problem = SVC_2d.SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)

    method_params = SolverParameters(r=np.double(3.0), itersLimit=100)
    solver = Solver(problem, parameters=method_params)

    apl = AnimationNDPaintListener("svc2d_anim.png", "output", varsIndxs=[0, 1], toPaintObjFunc=False)
    solver.AddListener(apl)

    spl = StaticNDPaintListener("svc2d_stat.png", "output", varsIndxs=[0, 1], mode="surface", calc="interpolation")
    solver.AddListener(spl)
    
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)

    solver_info = solver.Solve()

```

# **Структура проекта**

Последняя стабильная версия iOpt находится в [ветке main](https://github.com/UNN-ITMM-Software/iOpt/tree/main). 

Репозиторий включает в себя следующие директории::
- Пакет [iOpt](https://github.com/UNN-ITMM-Software/iOpt/tree/main/iOpt) содержит ядро фреймворка  в виде  классов на языке Python.
- Пакет [examples](https://github.com/UNN-ITMM-Software/iOpt/tree/main/examples) содержит примеры применения фреймворка для модельных и прикладных задач.
- Модульные тесты размещены в каталоге [test](https://github.com/UNN-ITMM-Software/iOpt/tree/main/test).
- Исходные файлы документации находятся в каталоге [docs](https://github.com/UNN-ITMM-Software/iOpt/tree/main/docs).

# **Документация**

Подробное описание API фреймворка iOpt доступно в разделе [Read the Docs](https://iopt.readthedocs.io/ru/latest/).

# **Поддержка**

Исследование проводится при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](https://sai.itmo.ru/) [Университета ИТМО](https://itmo.ru) в рамках мероприятия программы центра: Фреймворк методов интеллектуальной эвристической оптимизации.