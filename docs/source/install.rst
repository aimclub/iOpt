Установка и запуск
==================

Предварительные замечания
-------------------------

- В зависимости от текущей конфигурации, команда запуска Python может быть `python` или `python3`. В дальнейшем предполагается, что они обе запускают интерпретатор версии 3.
- Аналогично, команда запуска менджера пакетов может быть `pip` или `pip3`.
- Для построения документации необходимо предварительно установить пакет **sphinx** (лучше с правами администратора).
- Для установки и использования в ОС Windows нужно иметь настроенный интерпретатор команд (bash, cmd).

Автоматическая установка в Unix-подобных системах
--------------------------------------------------

Самый простой способ установить фреймворк:

- Сделать клон репозитория и перейти в его корневую папку

.. code-block::

    git clone https://github.com/UNN-ITMM-Software/iOpt
    cd iOpt


- Установить поддержку **virtualenv**

.. code-block::

    pip install virtualenv
    

- Создать и автивировать рабочее окружение **ioptenv**


.. code-block:: 

    virtualenv ioptenv
    source ioptenv/bin/activate


- Выполнить установку пакетов

.. code-block:: 

    python setup.py install


- Запустить примеры из папки **examples**

.. code-block:: 

    python examples/GKLS_example.py
    python examples/Rastrigin_example.py


- После окончания работы деактивировать виртуальное окружение

.. code-block:: 

    deactivate


Особенности автоматической установки в ОС Windows
-------------------------------------------------

После установки **virtualenv** и создания виртуального окружения, его активация осуществляется командой

.. code-block:: 

   ioptenv\Scripts\activate.bat


Ручная установка в Unix-подобных системах
-----------------------------------------

При этом способе необходимо:

- перейти в корень репозитория
- установить требуемые пакеты

.. code-block:: 

    pip install numpy depq cycler kiwisolver matplotlib scikit-learn sphinx sphinx_rtd_theme sphinxcontrib-details-directive  autodocsumm


- для доступа к модулю **iOpt** необходимо модифицировать переменную **PYTHONPATH** следующей командой

.. code-block:: 

    export PYTHONPATH="$PWD"
