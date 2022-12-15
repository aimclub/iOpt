<p align="center">
  <img src="https://github.com/UNN-ITMM-Software/iOpt/blob/main/docs/iOpt_logo.png" width="200" height="150"/>
</p>

iOpt - фреймворк с открытым исходным кодом для автоматического выбора значений параметров как для математических моделей сложных промышленных процессов, так и для используемых в промышленности методов ИИ и МО. Фреймворк распространяется под лицензией 3-Clause BSD.

[![License: BSD 3-Clause](https://img.shields.io/github/license/ITMO-NSS-team/Fedot.Industrial?style=flat-square)](https://github.com/UNN-ITMM-Software/iOpt/blob/main/LICENSE)

[![python: 3.9](https://img.shields.io/badge/python-3.9-green)](https://img.shields.io/badge/python-3.9-green)

[![python: 3.8](https://img.shields.io/badge/python-3.8-green)](https://img.shields.io/badge/python-3.8-green)




# Ключевые возможности фреймворка
- Автоматический выбор значений параметров математических моделей и методов ИИ и МО, используемых в промышленности.
- Интеллектуальное управление процессом выбора оптимальных параметров для промышленных задач.
- Интеграция с внешними библиотеками или фреймворками искусственного интеллекта и машинного обучения, а также предметными моделями.
- Автоматизация предварительного анализа исследуемых моделей, например, выделение различных классов зависимостей модели от разных групп параметров.
- Визуализация процесса выбора оптимальных параметров.

# Установка и настройка

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


# Начать работать

Использование фреймворка продемонстрируем на примере решения тестовой задачи с целевой функцией в виде параболоида.

```python
import math
import unittest
import sys
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener, AnimationPaintListener, StaticNDPaintListener, AnimationNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Запуск решения с визуализацией задачи Растригина с визуализацией
    """

    problem = Rastrigin(1)
    params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=True)
    solver = Solver(problem, parameters=params)

    pl = StaticPaintListener("rastrigin.png", "output", isPointsAtBottom = False)
    apl = AnimationPaintListener("rastriginAnim.png", "output", isPointsAtBottom = False, toPaintObjFunc=True)
    solver.AddListener(pl)
    solver.AddListener(apl)

    sol = solver.Solve()
    print(sol.numberOfGlobalTrials)
    print(sol.numberOfLocalTrials)
    print(sol.solvingTime)

    print(problem.knownOptimum[0].point.floatVariables)
    print(sol.bestTrials[0].point.floatVariables)
    print(sol.bestTrials[0].functionValues[0].value)
```

# Примеры использования

Описать пару примеров из папки example

# Структура проекта

Последняя стабильная версия фреймворка доступна в ветке [main](https://github.com/UNN-ITMM-Software/iOpt/tree/main).
Репозиторий включает следующие каталоги:
- Каталог [iOpt](https://github.com/UNN-ITMM-Software/iOpt/tree/main/iOpt) содержит ядро фреймворка в виде  классов на языке Python.
- Каталог [examples](https://github.com/UNN-ITMM-Software/iOpt/tree/main/examples) содержит примеры использования фреймворка как для решения модельных, так и прикладных задач.
- Модульные тесты размещены в каталоге [test](https://github.com/UNN-ITMM-Software/iOpt/tree/main/test).
- Исходные файлы документации находятся в каталоге [docs](https://github.com/UNN-ITMM-Software/iOpt/tree/main/docs).

# Документация

Детальное описание API фреймворка iOpt доступно на [Read the Docs](https://iopt.readthedocs.io/ru/latest/)
