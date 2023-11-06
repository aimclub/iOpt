Installation and how to use
==================

Preliminary remarks
-------------------------

- Depending on the current configuration, the Python startup command may be `python` or `python3`. In what follows, it is assumed that they both run the version 3 interpreter.
- Likewise, the command to start the package manager can be `pip` or `pip3`.
- To build documentation, you must first install the **sphinx** package (preferably with administrator privileges).
- To install and use the framework on Windows OS, you need to have a configured command interpreter (bash, cmd).

Automatic installation on Unix-like systems
--------------------------------------------------

The easiest way to install the framework:

- Clone the repository and go to its root folder

.. code-block::

    git clone https://github.com/aimclub/iOpt
    cd iOpt


- Install **virtualenv** support

.. code-block::

    pip install virtualenv
    

- Create and activate the working environment **ioptenv**

.. code-block:: 

    virtualenv ioptenv
    source ioptenv/bin/activate


- Install packages

.. code-block:: 

    python setup.py install


- To run examples from the **examples** folder

.. code-block:: 

    python examples/GKLS_example.py
    python examples/Rastrigin_example.py


- To deactivate the virtual environment after finishing work

.. code-block:: 

    deactivate


Differences in automatic installation in Windows OS
-------------------------------------------------

After installing **virtualenv** and creating a virtual environment, its activation is carried out with the command

.. code-block:: 

   ioptenv\Scripts\activate.bat


Manual installation on Unix-like systems
-----------------------------------------

In this case it is necessary:

- go to the root folder of the repository
- install the required packages

.. code-block:: 

    pip install numpy depq cycler kiwisolver matplotlib scikit-learn sphinx sphinx_rtd_theme sphinxcontrib-details-directive  autodocsumm


- to access the **iOpt** module you need to modify the **PYTHONPATH** variable with the following command

.. code-block:: 

    export PYTHONPATH="$PWD"
