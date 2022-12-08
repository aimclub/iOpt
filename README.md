<p align="center">
  <img src="https://github.com/UNN-ITMM-Software/iOpt/blob/main/docs/iOpt_logo.png" width="200" height="150"/>
</p>

iOpt - фреймворк с открытым исходным кодом для автоматического выбора значений параметров как для математических моделей сложных промышленных процессов, так и для используемых в промышленности методов ИИ и МО. Фреймворк распространяется под лицензией 3-Clause BSD.

# Ключевые возможности фреймворка
- Автоматический выбор значений параметров математических моделей и методов ИИ и МО, используемых в промышленности.
- Интеллектуальное управление процессом выбора оптимальных параметров для промышленных задач.
- Интеграция с внешними библиотеками или фреймворками искусственного интеллекта и машинного обучения, а также предметными моделями.
- Автоматизация предварительного анализа исследуемых моделей, например, выделение различных классов зависимостей модели от разных групп параметров.
- Визуализация процесса выбора оптимальных параметров.

# Установка и настройка

Описать, как "развернуть" фреймворк

# Начать работать

Использование фреймворка продемонстрируем на примере решения тестовой задачи с целевой функцией в виде параболоида.

```python
import unittest

from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

class TestSolveParaboloid(unittest.TestCase):
     def setUp(self):
         self.problem = XSquared(1)
         params = SolverParameters(r=3.5, eps=0.001)
         self.solver = Solver(self.problem, parameters=params)

     def test_solve(self):
         sol = self.solver.Solve()
         print(sol.bestTrials)
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

Детальное описание API фреймворка iOpt доступно по [ссылке]: (ссылка на сгенерированную документацию, см. как сделано в FEDOT)
