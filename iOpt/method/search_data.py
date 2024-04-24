from __future__ import annotations

import copy
import sys

import numpy as np
from depq import DEPQ

import json

from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.trial import Point, FunctionValue, FunctionType
from iOpt.trial import Trial


# from bintrees import AVLTree


class SearchDataItem(Trial):
    """
        The SearchDataItem class is intended for storing search information, which is an
        interval with a right point included, as well as links to neighbouring intervals. SearchDataItem
        is an inheritor of the Trial class
    """

    def __init__(self, y: Point, x: np.double,
                 function_values: np.ndarray(shape=(1), dtype=FunctionValue) = [FunctionValue()],
                 discrete_value_index: int = 0):
        """
        Constructor of SearchDataItem class

        :param y: trial point in the original N-dimensional search area.
        :param x: Mapping the trial point y to the segment [0, 1].
        :param function_values: Vector of function values (objective and constraint functions).
        :param discrete_value_index: Discrete parameter.
        """
        super().__init__(point=y, function_values=copy.deepcopy(function_values))
        self.point = y
        self.__x = x
        self.__discrete_value_index = discrete_value_index
        self.__index: int = -2
        self.__z: np.double = sys.float_info.max
        self.__leftPoint: SearchDataItem = None
        self.__rightPoint: SearchDataItem = None
        self.delta: np.double = -1.0
        self.globalR: np.double = -1.0
        self.localR: np.double = -1.0
        self.iterationNumber: int = -1
        self.blocked: bool = False
        self.creation_time = 0

    def get_x(self) -> np.double:
        """
        Obtain the right point of the search interval where :math:`x\in[0, 1]`

        :return: Value of the right point of the interval.
        """
        return self.__x

    def get_y(self) -> Point:
        """
        Provide an N-dimensional trial point of the original search area

        :return: N-dimensional trial point value.
        """
        return self.point

    def get_discrete_value_index(self) -> int:
        """
        Obtain a discrete parameter

        :return: Discrete parameter value.
        """
        return self.__discrete_value_index

    def set_index(self, index: int):
        """
        Specify the index value of the last executed constraint for the index scheme

        :param index: Restriction index.
        """
        self.__index = index

    def get_index(self) -> int:
        """
        Get the index value of the last executed constraint
        for the index scheme

        :return: Index value.
        """
        return self.__index

    def set_z(self, z: np.double):
        """
        Allow you to specify a function value for a given index.

        :param z: Function value.
        """
        self.__z = z

    def get_z(self) -> np.double:
        """
        Get the value of the function for a given index

        :return: Function value for index.
        """
        return self.__z

    def set_left(self, point: SearchDataItem):
        """
        Set the left interval for the source interval

        :param point: Left interval.
        """
        self.__leftPoint = point

    def get_left(self) -> SearchDataItem:
        """
        Get the left interval for the original interval

        :return: Left interval value.
        """
        return self.__leftPoint

    def set_right(self, point: SearchDataItem):
        """
        Set the right interval for the original interval

        :param point: Right interval.
        """
        self.__rightPoint = point

    def get_right(self) -> SearchDataItem:
        """
       Get the right interval for the original interval

       :return: Right interval value.
       """
        return self.__rightPoint

    def __lt__(self, other) -> bool:
        """
        The method overrides the < comparison operator for two intervals
        
        :param other: Second interval.
        :return: The value is true - if the right point of the initial interval is less than the
        the right point of the second interval, otherwise - false.
        """
        return self.get_x() < other.get_x()


class CharacteristicsQueue:
    """
    The CharacteristicsQueue class is designed to store a prioritised queue
    of characteristics with preempting
    """

    def __init__(self, maxlen: int):
        """
        Constructor of the CharacteristicsQueue class

        :param maxlen: Maximum queue size.
        """
        self.__baseQueue = DEPQ(iterable=None, maxlen=maxlen)

    def Clear(self):
        """
        Clear the queue
        """
        self.__baseQueue.clear()

    def insert(self, key: np.double, data_item: SearchDataItem):
        """
        Add search interval with specified priority.
        The priority is the value of the characteristic on this interval

        :param key: Priority of the search interval.
        :param data_item: Insertion interval.
        """
        self.__baseQueue.insert(data_item, key)

    def get_best_item(self) -> (SearchDataItem, np.double):
        """
        Get the interval with the best characteristic

        :return: Tuple: interval with the best characteristic, priority of the interval in the queue.
        """
        return self.__baseQueue.popfirst()

    def is_empty(self):
        """
        Check for queue emptiness.

        :return: True if the queue is empty, otherwise false.
        """
        return self.__baseQueue.is_empty()

    def get_max_len(self) -> int:
        """
        Get the maximum queue size.

        :return: Value of maximum queue size.
        """
        return self.__baseQueue.maxlen

    def get_len(self) -> int:
        """
        Get the current queue size

        :return: Value of the current queue size.
        """
        return len(self.__baseQueue)


class SearchData:
    """
    The SearchData class is used to store the set of all intervals, the original task
    and the priority queue of global characteristics
    """

    # очереди характеристик
    # _RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue(None)
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    # _allTrials: List = []
    # __firstDataItem:

    # solution: Solution = None

    def __init__(self, problem: Problem, maxlen: int = None):
        """
        Constructor of SearchData class

        :param problem: Information about the original task.
        :param maxlen: Maximum queue size.
        """
        self.solution = Solution(problem)
        self._allTrials = []
        self._RGlobalQueue = CharacteristicsQueue(maxlen)
        self.__firstDataItem: SearchDataItem = None

    def clear_queue(self):
        """
        Clear the characteristic queue
        """
        self._RGlobalQueue.Clear()

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если right_data_item == None то его необходимо найти по дереву _allTrials
    def insert_data_item(self, new_data_item: SearchDataItem,
                         right_data_item: SearchDataItem = None):
        """
        Add a new trial interval to the list of all trials performed
        and prioritised characteristic queue

        :param new_data_item: New trial interval.
        :param right_data_item: The covering interval, is the right interval for the newDataItem.
        """
        flag = True
        if right_data_item is None:
            right_data_item = self.find_data_item_by_one_dimensional_point(new_data_item.get_x())
            flag = False
        new_data_item.set_left(right_data_item.get_left())
        right_data_item.set_left(new_data_item)
        new_data_item.set_right(right_data_item)
        new_data_item.get_left().set_right(new_data_item)

        self._allTrials.append(new_data_item)

        self._RGlobalQueue.insert(new_data_item.globalR, new_data_item)
        if flag:
            self._RGlobalQueue.insert(right_data_item.globalR, right_data_item)

    def insert_first_data_item(self, left_data_item: SearchDataItem,
                               right_data_item: SearchDataItem):
        """
        Allow a pair of trial intervals to be added to the first iteration of the GSA.

        :param left_data_item: Left interval for right_data_item.
        :param right_data_item: Right interval for left_data_item.
        """
        left_data_item.set_right(right_data_item)
        right_data_item.set_left(left_data_item)

        self._allTrials.append(left_data_item)
        self._allTrials.append(right_data_item)

        self.__firstDataItem = left_data_item

    # поиск покрывающего интервала
    # возвращает правую точку
    def find_data_item_by_one_dimensional_point(self, x: np.double) -> SearchDataItem:
        """
        Find the covering interval for the obtained point x

        :param x: Right point of the interval.
        :return: Right point of the covering interval.
        """
        # итерируемся по rightPoint от минимального элемента
        for item in self:
            if item.get_x() > x:
                return item
        return None

    def get_data_item_with_max_global_r(self) -> SearchDataItem:
        """
        Obtain the interval with the best value of the global characteristic

        :return: Value of the interval with the best global characteristic.
        """
        if self._RGlobalQueue.is_empty():
            self.refill_queue()
        return self._RGlobalQueue.get_best_item()[0]

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def refill_queue(self):
        """
        Refill the queue of global characteristics, for example, when it is empty
        or when the Lipschitz constant estimation is changed

        """
        self._RGlobalQueue.Clear()
        for itr in self:
            if not itr.blocked:
                self._RGlobalQueue.insert(itr.globalR, itr)

    # Возвращает текущее число интервалов в дереве
    def get_count(self) -> int:
        """
        Get the current number of intervals in the list

        :return: Value of the number of intervals in the list.
        """
        return len(self._allTrials)

    def get_last_item(self) -> SearchDataItem:
        """
        Get the last added interval to the list

        :return: Value of the last added interval.
        """
        try:
            return self._allTrials[-1]
        except Exception:
            print("GetLastItem: List is empty")

    def get_last_items(self, N: int = 1) -> list[SearchDataItem]:
        """
        Get the last added intervals to the list.

        :return: Values of the last series of added intervals.
        """
        try:
            return self._allTrials[-N:]
        except Exception:
            print("GetLastItems: List is empty")

    def searchdata_to_json(self, mode ='full') -> json: #что именно возвращает? По идее просто словарь с кучей вложений, но насколько понятно?
        """
        Save the optimization process to a file

        :param mode: 'full' - save all optimization information
        :param file_name: file name.
        """
        data = {}
        data['SearchDataItem'] = []
        for dataItem in self._allTrials:
            fvs = []
            for fv in dataItem.function_values:
                fvs.append({
                    'value': fv.value,
                    'type': 1 if fv.type == FunctionType.OBJECTIV else 2,
                    'functionID': str(fv.functionID),
                })

            if np.isfinite(dataItem.get_z()):
                data['SearchDataItem'].append({
                    'float_variables': list(dataItem.get_y().float_variables),
                    'discrete_variables': [] if dataItem.get_y().discrete_variables is None else list(
                        dataItem.get_y().discrete_variables),
                    'function_values': list(fvs),
                    'x': dataItem.get_x(),
                    'delta': dataItem.delta,
                    'globalR': dataItem.globalR,
                    'localR': dataItem.localR,
                    'index': dataItem.get_index(),
                    'discrete_value_index': dataItem.get_discrete_value_index(),
                    '__z': dataItem.get_z(),
                    'creation_time': dataItem.creation_time,
                    'iterationNumber': dataItem.iterationNumber
                })

        num_iterations_best = [] #в случае с mco - несколько лучших
        data['best_trials'] = []
        for dataItem in self.solution.best_trials: # сохранение всех лучших (если несколько, например, в mco)
            for fv in dataItem.function_values:
                fvs.append({
                    'value': fv.value,
                    'type': 1 if fv.type == FunctionType.OBJECTIV else 2,
                    'functionID': str(fv.functionID),
                })

            data['best_trials'].append({
                'float_variables': list(dataItem.get_y().float_variables),
                'discrete_variables': [] if dataItem.get_y().discrete_variables is None else list(
                    dataItem.get_y().discrete_variables),
                'function_values': list(fvs),
                'x': dataItem.get_x(),
                'delta': dataItem.delta,
                'globalR': dataItem.globalR,
                'localR': dataItem.localR,
                'index': dataItem.get_index(),
                'discrete_value_index': dataItem.get_discrete_value_index(),
                '__z': dataItem.get_z(),
                'creation_time': dataItem.creation_time,
                'iterationNumber': dataItem.iterationNumber
            })

            num_iterations_best.append(dataItem.iterationNumber)

        if mode == 'full':
            data['solution'] = []
            data['solution'].append({
                'number_of_trials': (self.solution.number_of_global_trials + self.solution.number_of_local_trials),
                'number_of_global_trials': self.solution.number_of_global_trials,
                'number_of_local_trials': self.solution.number_of_local_trials,
                'solving_time': self.solution.solving_time,
                'solution_accuracy': self.solution.solution_accuracy,
                'num_iteration_best_trial': list(num_iterations_best)
            })

            float_variables = []
            for i in range(self.solution.problem.number_of_float_variables):
                bounds = [self.solution.problem.lower_bound_of_float_variables[i],
                          self.solution.problem.upper_bound_of_float_variables[i]]
                float_variables.append({
                    str(self.solution.problem.float_variable_names[i]): (list(bounds)),
                })

            discrete_variables = []
            for i in range(self.solution.problem.number_of_discrete_variables):
                discrete_variables.append({
                    str(self.solution.problem.discrete_variable_names[i]):
                        (list(self.solution.problem.discrete_variable_values[i])),
                })

            data['Task'] = []
            data['Task'].append({
                'float_variables': list(float_variables),
                'discrete_variables': list(discrete_variables),
                'name': self.solution.problem.name
            })

        return data

    def json_to_searchdata(self, data, mode ='full'):
        """
        Load the optimization process from a file

        :param file_name: file name.
        """
        function_values = []
        for trial in data['best_trials']:
            for fv in trial['function_values']:
                function_values.append(FunctionValue(
                    (FunctionType.OBJECTIV if fv['type'] == 1 else FunctionType.CONSTRAINT),
                    str(fv['functionID'])))
                function_values[-1].value = np.double(fv['value'])

            data_item = SearchDataItem(Point(trial['float_variables'], trial['discrete_variables']), trial['x'],
                                       function_values,
                                       trial['discrete_value_index'])
            data_item.delta = trial['delta']
            data_item.globalR = trial['globalR']
            data_item.localR = trial['localR']
            data_item.set_z(trial['__z'])
            data_item.set_index(trial['index'])

            self.solution.best_trials[0] = data_item
            if mode == 'only search_data':
                self.solution.solution_accuracy = min(data_item.delta, self.solution.solution_accuracy)

        first_data_item = []
        for trial in data['SearchDataItem'][:2]:
            function_values = []
            for fv in trial['function_values']:
                function_values.append(FunctionValue(
                    (FunctionType.OBJECTIV if fv['type'] == 1 else FunctionType.CONSTRAINT),
                    str(fv['functionID'])))
                function_values[-1].value = np.double(fv['value'])

            first_data_item.append(
                SearchDataItem(Point(trial['float_variables'], trial['discrete_variables']), trial['x'], function_values,
                               trial['discrete_value_index']))
            first_data_item[-1].delta = trial['delta']
            first_data_item[-1].globalR = trial['globalR']
            first_data_item[-1].localR = trial['localR']
            first_data_item[-1].set_index(trial['index'])

        self.insert_first_data_item(first_data_item[0], first_data_item[1])

        for trial in data['SearchDataItem'][2:]:
            function_values = []
            for fv in trial['function_values']:
                function_values.append(FunctionValue(
                    (FunctionType.OBJECTIV if fv['type'] == 1 else FunctionType.CONSTRAINT),
                    str(fv['functionID'])))
                function_values[-1].value = np.double(fv['value'])

            data_item = SearchDataItem(Point(trial['float_variables'], trial['discrete_variables']),
                                       trial['x'], function_values, trial['discrete_value_index'])
            data_item.delta = trial['delta']
            data_item.globalR = trial['globalR']
            data_item.localR = trial['localR']
            data_item.set_z(trial['__z'])
            data_item.creation_time = trial['creation_time']
            data_item.set_index(trial['index'])

            self.insert_data_item(data_item)

        if mode == 'full':
            for trial in data['solution']:
                self.solution.number_of_global_trials = trial['number_of_global_trials']
                self.solution.number_of_local_trials = trial['number_of_local_trials']
                self.solution.solving_time = trial['solving_time']
                self.solution.solution_accuracy = trial['solution_accuracy']

    def __iter__(self):
        # вернуть самую левую точку из дерева (ниже код проверить!)
        # return self._allTrials.min_item()[1]
        self.curIter = self.__firstDataItem
        if self.curIter is None:
            raise StopIteration
        else:
            return self

    def __next__(self):
        if self.curIter is None:
            raise StopIteration
        else:
            tmp = self.curIter
            self.curIter = self.curIter.get_right()
            return tmp


class SearchDataDualQueue(SearchData):
    """
    The SearchDataDualQueue class is incherited of the SearchData class. It is intended
      for storing a set of all intervals, the initial task and two priority queues
      for global and local characteristics

    """

    def __init__(self, problem: Problem, maxlen: int = None):
        """
        Constructor of SearchDataDualQueue class

        :param problem: Information on the initial task.
        :param maxlen: Maximum queue size.
        """
        super().__init__(problem, maxlen)
        self.__RLocalQueue = CharacteristicsQueue(maxlen)

    def clear_queue(self):
        """
        Clear the characteristic queues
        """
        self._RGlobalQueue.Clear()
        self.__RLocalQueue.Clear()

    def insert_data_item(self, new_data_item: SearchDataItem,
                         right_data_item: SearchDataItem = None):
        """
        Add a new trial interval to the list of all trials performed
          and priority queues of global and local characteristics

        :param new_data_item: New trial interval.
        :param right_data_item: The covering interval, is the right interval for the new_data_item.
        """
        flag = True
        if right_data_item is None:
            right_data_item = self.find_data_item_by_one_dimensional_point(new_data_item.get_x())
            flag = False

        new_data_item.set_left(right_data_item.get_left())
        right_data_item.set_left(new_data_item)
        new_data_item.set_right(right_data_item)
        new_data_item.get_left().set_right(new_data_item)

        self._allTrials.append(new_data_item)

        self._RGlobalQueue.insert(new_data_item.globalR, new_data_item)
        self.__RLocalQueue.insert(new_data_item.localR, new_data_item)
        if flag:
            self._RGlobalQueue.insert(right_data_item.globalR, right_data_item)
            self.__RLocalQueue.insert(right_data_item.localR, right_data_item)

    def get_data_item_with_max_global_r(self) -> SearchDataItem:
        """
       Obtain the interval with the best value of the global characteristic

       :return: Value of the interval with the best global characteristic.
       """
        if self._RGlobalQueue.is_empty():
            self.refill_queue()
        best_item = self._RGlobalQueue.get_best_item()
        while best_item[1] != best_item[0].globalR:
            if self._RGlobalQueue.is_empty():
                self.refill_queue()
            best_item = self._RGlobalQueue.get_best_item()
        return best_item[0]

    def get_data_item_with_max_local_r(self) -> SearchDataItem:
        """
       Obtain the interval with the best value of the local characteristic

       :return: Value of the interval with the best local characteristic.
       """
        if self.__RLocalQueue.is_empty():
            self.refill_queue()
        best_item = self.__RLocalQueue.get_best_item()
        while best_item[1] != best_item[0].localR:
            if self.__RLocalQueue.is_empty():
                self.refill_queue()
            best_item = self.__RLocalQueue.get_best_item()
        return best_item[0]

    def refill_queue(self):
        """
       Refill the queues of global and local characteristics, e.g.,
         when they are emptied or when the Lipschitz constant estimation is changed

       """
        self.clear_queue()
        for itr in self:
            self._RGlobalQueue.insert(itr.globalR, itr)
            self.__RLocalQueue.insert(itr.localR, itr)
