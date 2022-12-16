<p align="center">
  <img src="https://github.com/UNN-ITMM-Software/iOpt/blob/main/docs/iOpt_logo.png" width="200" height="150"/>
</p>

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-green)](https://github.com/UNN-ITMM-Software/iOpt/blob/main/LICENSE)
[![python: 3.9](https://img.shields.io/badge/python-3.9-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-390/)
[![python: 3.8](https://img.shields.io/badge/python-3.8-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-380/)
[![docs: ](https://readthedocs.org/projects/ebonite/badge/?style=flat-square)](https://iopt.readthedocs.io/ru/latest/)
[![build:](https://github.com/UNN-ITMM-Software/iOpt/actions/workflows/python-app.yml/badge.svg)](https://github.com/UNN-ITMM-Software/iOpt/actions)



iOpt - фреймворк с открытым исходным кодом для автоматического выбора значений параметров как для математических моделей сложных промышленных процессов, так и для используемых в промышленности методов ИИ и МО. Фреймворк распространяется под лицензией 3-Clause BSD.


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

Использование фреймворка iOpt для минимизации функции Растригина.

```python
from iOpt.problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Минимизация тестовой функции Растригина с визуализацией
    """
    #Создание тестовой задачи
    problem = Rastrigin(2)
    #Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)
    #Создание решателя
    solver = Solver(problem, parameters=params)
    #Вывод результатов в консоль в процессе решения
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)
    #3D визуализация по окончании решения
    spl = StaticNDPaintListener("rastrigin.png", "output", varsIndxs=[0,1], mode="surface", calc="interpolation")
    solver.AddListener(spl)
    #Запуск решения задачи
    sol = solver.Solve()
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
