<p align="center">
  <img src="/docs/iOpt_logo.png" width="200" height="150"/>
</p>

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-green)](https://github.com/aimclub/iOpt/blob/main/LICENSE)
[![python: 3.9](https://img.shields.io/badge/python-3.9-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-390/)
[![python: 3.8](https://img.shields.io/badge/python-3.8-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-380/)
[![docs: ](https://readthedocs.org/projects/ebonite/badge/?style=flat-square)](https://iopt.readthedocs.io/ru/latest/)
[![build:](https://github.com/UNN-ITMM-Software/iOpt/actions/workflows/python-app.yml/badge.svg)](https://github.com/aimclub/iOpt/actions)
[![eng:](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/aimclub/iOpt/blob/main/README.md)



iOpt -  это фреймворк с открытым исходным кодом. iOpt предназначен для автоматического подбора параметров как для 
математических моделей сложных промышленных процессов, так и для моделей AI и ML.
Фреймворк распространяется под лицензией 3-Clause BSD.

# **Основные возможности**
- Автоматический подбор параметров как для математических моделей, так и для моделей AI и ML, используемых в промышленности.
- Интеллектуальный контроль процесса выбора оптимальных параметров для промышленного применения.
- Интеграция с библиотеками и фреймворками искусственного интеллекта и машинного обучения, а также с прикладными моделями.
- Автоматизация предварительного анализа исследуемых моделей, например, путем выявления различных видов зависимостей модели от разных групп параметров.
- Визуализация процесса поиска оптимальных параметров.


# **Установка**

## Unix-like системы:

```
git clone https://github.com/UNN-ITMM-Software/iOpt
cd iOpt
pip install virtualenv
virtualenv ioptenv
source ioptenv/bin/activate
python setup.py install
```

## Windows:

```
git clone https://github.com/UNN-ITMM-Software/iOpt
cd iOpt
pip install virtualenv
virtualenv ioptenv
ioptenv\Scripts\activate.bat
python setup.py install
```


# **Как использовать**

Минимизация функции Растригина с помощью iOpt.

```python
from iOpt.problems.rastrigin import Rastrigin
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
    #Создание оптимизатора
    solver = Solver(problem, parameters=params)
    #Вывод результатов на консоль во время решения
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)
    #3D визуализация после получения решения
    spl = StaticNDPaintListener("rastrigin.png", "output", varsIndxs=[0,1], mode="surface", calc="interpolation")
    solver.AddListener(spl)
    #Запуск оптимизации
    sol = solver.Solve()
```

# **Примеры**

Продемонстрируем как использовать iOpt для настройки гиперпараметров модели машинного обучения.
Для метода опорных векторов ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)) найдем оптимальные 
гиперпараметры (**C** - параметр регуляризации, **gamma** - коэффициент ядра) для решения задачи классификации рака молочной железы
 ([описание датасета](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))).


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
- Пакет [iOpt](https://github.com/UNN-ITMM-Software/iOpt/tree/main/iOpt) содержит ядро фреймворка и состоит из его основных классов.
- Пакет [examples](https://github.com/UNN-ITMM-Software/iOpt/tree/main/examples) содержит примеры применения фреймворка для тестовых и прикладных задач.
- Модульные тесты находятся в каталоге [test](https://github.com/UNN-ITMM-Software/iOpt/tree/main/test).
- Источники документации находятся в каталоге [docs](https://github.com/UNN-ITMM-Software/iOpt/tree/main/docs).

# **Документация**

Подробное описание iOpt API доступно в разделе [Read the Docs](https://iopt.readthedocs.io/ru/latest/).
