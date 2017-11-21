"""
.. module:: eliana_test
    :synopsis: base module for tests

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 12, 2017
"""

import os
from abc import ABC, abstractmethod
import sys


class ElianaTest(ABC):
    """.. class:: ElianaTest

    Base module for all types of tests.
    """
    def __init__(self):
        self.dir_working = os.getcwd()

        print(sys.version, '\n')

        self.__dir_env_modules = os.path.join(
            self.dir_working,
            'env',
            'eliana',
            'lib',
            'python3.6',
            'site-packages'
        )
        self.__dir_local_modules = os.path.join(
            self.dir_working,
            'lib'
        )
        self.training_data = os.path.join(
            self.dir_working,
            'training'
        )
        self.eliana_log = ElianaLog()

    @property
    def dir_env_modules(self):
        return self.__dir_env_modules

    @dir_env_modules.setter
    def dir_env_modules(self, dir_):
        self.__dir_env_modules = dir_

    @property
    def dir_local_modules(self):
        return self.__dir_local_modules

    @dir_local_modules.setter
    def dir_local_module(self, dir_):
        self.__dir_local_modules = dir_

    @abstractmethod
    def run(self):
        pass

    def test(self, func):
        try:
            func()
        except Exception:
            print(self.eliana_log.log_error, '\n')
            traceback.print_exc()
            print(
                '\n' +
                str(self.eliana_log.step_counter),
                'out of',
                str(self.eliana_log.steps),
                'steps executed. Exiting...',
            )
            exit()
        else:
            print(self.eliana_log.log_ok)


class ElianaUnitTest(ElianaTest):
    """.. class:: ElianaUnitTest

    Base module for unit tests.
    """
    def __init__(self):
        ElianaTest.__init__(self)


class ElianaIntegratedtTest(ElianaTest):
    """.. class:: ElianaIntegratedTest

    Base module for integrated tests.
    """
    def __init__(self):
        ElianaTest.__init__(self)


class ElianaLog:

    # TODO:
    # Use logger class in here.

    def __init__(self):
        self.__log_ok = self.__status_formatter('OK')
        self.__log_error = self.__status_formatter('ERROR')
        self.__log_warning = self.__status_formatter('WARNING')

        self.__steps = 0
        self.__step_counter = 1

    @property
    def step_counter(self):
        return self.__step_counter

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, steps):
        self.__steps = steps

    @property
    def log_ok(self):
        return self.__log_ok

    @property
    def log_error(self):
        return self.__log_error

    @property
    def log_warning(self):
        return self.__log_warning

    def __status_formatter(self, log_string) -> str:
        return ' [' + log_string + ']'

    def log(self, log_string):
        print(
            log_string,
            ' (',
            self.__step_counter,
            '/',
            self.__steps,
            ')',
            '...',

            end='',
            sep=''
        )
        self.__step_counter += 1
