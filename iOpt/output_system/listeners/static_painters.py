from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.search_data import SearchData
from iOpt.output_system.painters.static_painters import StaticPainter, StaticPainterND, DiscretePainter
from iOpt.solution import Solution


class StaticDiscreteListener(Listener):
    """
    """

    def __init__(self, file_name: str, path_for_saves="", mode='analysis', calc='objective function',
                 type='lines layers', numpoints=150, mrkrs=3):
        """
        """
        if mode != 'analysis' and mode != 'bestcombination':
            raise Exception(
                "StaticDiscreteListener mode is incorrect, mode can take values 'analysis' or 'bestcombination'")
        if type != 'lines layers':
            raise Exception(
                "StaticDiscreteListener type is incorrect, type can take values 'lines layers'")
        if calc != 'objective function' and calc != 'interpolation':
            raise Exception(
                "StaticDiscreteListener calc is incorrect, calc can take values 'objective function' or 'interpolation'")
        if numpoints <= 0:
            raise Exception(
                "StaticDiscreteListener numpoints is incorrect, numpoints > 0")
        if mrkrs <= 0:
            raise Exception(
                "StaticDiscreteListener mrkrs is incorrect, mrkrs > 0")

        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.subparameters = [1, 2]
        self.mode = mode
        self.type = type
        self.calc = calc
        self.numpoints = numpoints
        self.mrkrs = mrkrs
        self.search_dataSorted = []
        self.bestValueSorted = []
        self.number_of_parallel_points = 1

    def before_method_start(self, method: Method):
        if method.task.problem.number_of_float_variables > 2 and self.calc == 'interpolation':
            raise Exception(
                "StaticDiscreteListener with calc 'interpolation' supported with dimension <= 2")
        self.number_of_parallel_points = method.parameters.number_of_parallel_points

    def on_end_iteration(self, new_points, solution: Solution):
        for newPoint in new_points:
            self.search_dataSorted.append(newPoint)
            self.bestValueSorted.append(solution.best_trials[0].function_values[0].value)

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool):
        painter = DiscretePainter(self.search_dataSorted, self.bestValueSorted,
                                  solution.problem.number_of_discrete_variables,
                                  solution.problem.number_of_float_variables,
                                  solution.best_trials[0].point,
                                  solution.problem.discrete_variable_values,
                                  solution.problem.discrete_variable_names,
                                  self.mode, self.calc, self.subparameters,
                                  solution.problem.lower_bound_of_float_variables,
                                  solution.problem.upper_bound_of_float_variables,
                                  self.file_name, self.path_for_saves, solution.problem.calculate,
                                  solution.best_trials[0].function_values[0].value,
                                  search_data, self.number_of_parallel_points)
        if self.mode == 'analysis':
            painter.paint_analisys(mrks=2)
        elif self.mode == 'bestcombination':
            if self.type == 'lines layers':
                painter.paint_objective_func(self.numpoints)
                painter.paint_points(self.mrkrs)

        painter.save_image()


# mode: objective function, approximation, only points
class StaticPainterListener(Listener):
    """
    Класс StaticPaintListener - слушатель событий. Содержит метод-обработчик, выдающий в качестве
      реакции на завершение работы метода изображение.
    """

    def __init__(self, file_name: str, path_for_saves="", indx=0, is_points_at_bottom=False, mode='objective function'):
        """
        Конструктор класса StaticPaintListener

        :param file_name: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param path_for_saves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param indx: Индекс переменной оптимизационной задачи. Используется в многомерной оптимизации.
           Позволяет отобразить в сечении найденного минимума процесс оптимизации по одной выбранной переменной.
        :param is_points_at_bottom: Отрисовать точки поисковой информации под графиком. Если False, точки ставятся на графике.
        :param mode: Способ вычислений для отрисовки графика целевой функции, который будет использован. Возможные
           режимы: 'objective function', 'only points', 'approximation' и 'interpolation'. Режим 'objective function'
           строит график, вычисляя значения целевой функции на равномерной сетке. Режим 'approximation' строит
           нейроаппроксимацию для целевой функции на основе полученной поисковой информации.
           Режим 'interpolation' строит интерполяцию для целевой функции на основе полученной поисковой информации.
           Режим 'only points' не строит график целевой функции.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.parameterInNDProblem = indx
        self.is_points_at_bottom = is_points_at_bottom
        self.mode = mode

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool):
        painter = StaticPainter(search_data, solution, self.mode, self.is_points_at_bottom,
                                self.parameterInNDProblem, self.path_for_saves, self.file_name)
        painter.paint_objective_func()
        painter.paint_points()
        painter.paint_optimum()
        painter.save_image()


# mode: surface, lines layers, approximation
class StaticPainterNDListener(Listener):
    """
    Класс StaticNDPaintListener - слушатель событий. Содержит метод-обработчик, выдающий в качестве
      реакции на завершение работы метода изображение.
      Используется для многомерной оптимизации.
    """

    def __init__(self, file_name: str, path_for_saves="", vars_indxs=[0, 1], mode='lines layers',
                 calc='objective function'):
        """
        Конструктор класса StaticNDPaintListener

        :param file_name: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param path_for_saves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param vars_indxs: Пара индексов переменных оптимизационной задачи, для которых будет построен рисунок.
        :param mode_: Режим отрисовки графика целевой функции, который будет использован.
           Возможные режимы:'lines layers', 'surface'.
           Режим 'lines layers' рисует линии уровня в сечении найденного методом решения.
           Режим 'surface' строит поверхность в сечении найденного методом решения.
        :param calc: Способ вычислений для отрисовки графика целевой функции, который будет использован. Возможные
           режимы: 'objective function' (только в режиме 'lines layers'), 'approximation' (только в режиме 'surface')
           и 'interpolation'. Режим 'objective function' строит график, вычисляя значения целевой функции на равномерной
           сетке. Режим 'approximation' строит нейроаппроксимацию для целевой функции на основе полученной поисковой
           информации. Режим 'interpolation' строит интерполяцию для целевой функции на основе полученной поисковой
           информации.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.parameters = vars_indxs
        self.mode = mode
        self.calc = calc

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool, ):
        painter = StaticPainterND(search_data, solution, self.parameters, self.mode, self.calc,
                                  self.file_name, self.path_for_saves)
        painter.paint_objective_func()
        painter.paint_points()
        painter.paint_optimum()
        painter.save_image()
