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
import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.rastrigin import Rastrigin

class TestRastrigin(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""
    def setUp(self):
        self.rastrigin = Rastrigin(3)

    def test_Calculate(self):
        point = Point([1.0, 0.5, 0.3], [])
        sum: np.double = 0
        for i in range(self.rastrigin.dimension):
            sum += point.floatVariables[i] * point.floatVariables[i] - 10 * math.cos(
                2 * math.pi * point.floatVariables[i]) + 10

        functionValue = FunctionValue()
        functionValue = self.rastrigin.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, sum)

    def test_OptimumValue(self):
        self.assertEqual(self.rastrigin.knownOptimum[0].functionValues[0].value, 0.0)

"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
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

Детальное описание API фреймворка iOpt доступно по [ссылке]: (ссылка на сгенерированную документацию)
