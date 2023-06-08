from abc import ABC, abstractmethod
import math
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt


class RandomVariable(ABC):
    """Абстрактный базовый класс для случайных величин."""

    @abstractmethod
    def pdf(self, x):
        """
        Функция плотности вероятности (PDF) случайной величины.

        Args:
            x: Значение для оценки PDF.

        Returns:
            Значение PDF при заданном значении.
        """

        pass

    @abstractmethod
    def cdf(self, x):
        """
        Кумулятивная функция распределения (CDF) случайной величины.

        Args:
            x: Значение, при котором оценивается CDF.

        Returns:
            Значение PDF при заданном значении.
        """
        pass

    @abstractmethod
    def quantile(self, alpha):
        """
        Квантильная функция случайной величины.

        Args:
            alpha: Значение вероятности.

        Returns:
            Значение квантиля, соответствующее заданной вероятности.
        """
        pass


class LaplaceDistribution(RandomVariable):
    """Случайная величина распределения Лапласа."""

    def __init__(self, location, scale):
        """
        Инициализация распределения Лапласа с указанным местоположением и параметрами масштаба.

        Args:
            location: Параметр масштаба.
            scale: Параметр сдвига.
        """
        self.location = location
        self.scale = scale

    def pdf(self, x):
        """
        Функция плотности вероятности (PDF) распределения Лапласа.

        Args:
            x: Значение для оценки PDF

        Returns:
            Значение PDF при заданном значении.
        """
        return (1 / (2 * self.scale)) * np.exp(-np.abs(x - self.location) / self.scale)

    def cdf(self, x):
        """
        Кумулятивная функция распределения (CDF) распределения Лапласа.

        Args:
            x: Значение, при котором оценивается CDF.

        Returns:
            Значение CDF при заданном значении.
        """
        if x < self.location:
            return 0.5 * np.exp((x - self.location) / self.scale)
        else:
            return 1 - 0.5 * np.exp(-(x - self.location) / self.scale)

    def quantile(self, alpha):
        """
        Квантильная функция распределения Лапласа.

        Args:
            alpha: Значение вероятности.

        Returns:
            Значения квантиля, соответствующее заданной вероятности.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Численное значение квантиля может быть в промежутке между 0 и 1.")
        if alpha == 0.5:
            return self.location
        elif alpha < 0.5:
            return self.location + self.scale * np.log(2 * alpha)
        else:
            return self.location - self.scale * np.log(2 * (1 - alpha))


class NonParametricRandomVariable(RandomVariable):
    """Непараметрическая случайная величина."""

    def __init__(self, source_sample) -> None:
        """
        Инициализация непараметрической случайной величины с данной исходной выборкой.

        Args:
            source_sample: Исходная выборка для непараметрической случайной величины.
        """
        super().__init__()
        self.source_sample = sorted(source_sample)

    def pdf(self, x):
        """
        Функция плотности вероятности (PDF) непараметрической случайной величины.

        Args:
            x: Значение для оценки PDF.

        Returns:
            Значение PDF при заданном значении.
        """
        if x in self.source_sample:
            return float('inf')
        return 0

    @staticmethod
    def heaviside_function(x):
        """
        Функция Хевисайда.

        Args:
            x: Входное значение.

        Returns:
            Значении функции Хевисайда на данном входе.
        """
        if x > 0:
            return 1
        else:
            return 0

    def cdf(self, x):
        """
        Кумулятивная функция распределения (CDF) непараметрической случайной величины.

        Args:
            x: Значение, при котором оценивается CDF.

        Returns:
            Значение CDF при заданном значении.
        """
        return np.mean(np.vectorize(self.heaviside_function)(x - self.source_sample))

    def quantile(self, alpha):
        """
        Квантильная функция непараметрической случайной величины.

        Args:
            alpha: Значение вероятности.

        Returns:
            Значение квантиля, соответствующее заданной вероятности.
        """
        index = int(alpha * len(self.source_sample))
        return self.source_sample[index]


class RandomNumberGenerator(ABC):
    """Абстрактный базовый класс для генераторов случайных чисел."""

    def __init__(self, random_variable: RandomVariable):
        """
        Инициализация генератора случайных чисел с указанной случайной величиной.

        Args:
            random_variable: Случайная переменная, используемая для генерации случайных чисел.
        """
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        """
        Генерация случайных чисел.

        Args:
            N: Количество случайных чисел для генерации.

        Returns:
            Массив сгенерированных случайных чисел.
        """
        pass


class SimpleRandomNumberGenerator(RandomNumberGenerator):
    """Простой генератор случайных чисел."""

    def __init__(self, random_variable: RandomVariable):
        """
        Инициализация простого генератора случайных чисел с указанной случайной величиной.

        Args:
            random_variable: Случайная переменная, используемая для генерации случайных чисел.
        """
        super().__init__(random_variable)

    def get(self, N):
        """
        Генерация случайных величин с помощью простого генератора случайных величин.

        Args:
            N: Количество случайных чисел для генерации.

        Returns:
            Массив сгенерированных случайных чисел.
        """
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)


class TukeyRandomNumberGenerator(RandomNumberGenerator):
    """Генератор случайных чисел Тьюки."""
    def __init__(self, sample, location, scale, proc):
        """
        Инициализация генератора случайных чисел Тьюки с указанными параметрами.

        Args:
            sample: Оригинальная выборка.
            location: Параметр масштаба.
            scale: Параметр сдвига.
            proc: Доля выбросов.
        """
        self.data = sample
        self.location = location
        self.scale = scale
        self.proc = proc

    def get(self, N):
        """
        Генерация случайных чисел с помощью генератора случайных чисел Тьюки.

        Args:
            N: Количество случайных чисел для генерации.

        Returns:
            Массив сгенерированных случайных чисел.
        """
        proc_count = int(self.proc * N)
        outliersRV = LaplaceDistribution(self.location, self.scale)
        generator = SimpleRandomNumberGenerator(outliersRV)
        outliers = generator.get(proc_count)
        data_proc = np.concatenate((self.data, outliers))
        return data_proc


class Estimation(ABC):
    """Абстрактный базовый класс для оценок."""
    @abstractmethod
    def estimate(self, sample):
        """
        Оценка параметра или характеристики на основе данной выборки.

        Args:
            sample: Выборка, используемая для оценки.

        Returns:
            Значение оценки.
        """
        pass


class HodgesLehmannEstimation(Estimation):
    """Оценка Ходжеса-Лемана."""
    def estimate(self, sample):
        """
        Оценка параметра масштаба с помощью оценки Ходжеса-Лемана.

        Args:
            sample: Выборка, используемая для оценки.

        Returns:
            Оценка параметра масштаба.
        """
        n = len(sample)
        means = [(sample[i] + sample[j]) / 2 for i in range(n) for j in range(i, n)]
        return np.median(means)


class HalfSumEstimation(Estimation):
    """Оценка полусуммы порядковых статистик."""
    def estimate(self, sample):
        """
        Оценка параметра масштаба с помощью оценки полусуммы порядковых статистик.

        Args:
            sample: Выборка, используемая для оценки.

        Returns:
            Оценка параметра масштаба.
        """
        sorted_sample = sorted(sample)
        n = len(sorted_sample)
        r = 0.5 * n


        if r.is_integer():
            r = int(r)
            half_sum = 0.5 * (sorted_sample[r-1] + sorted_sample[n-r])
        else:
            r_floor = int(r // 1)
            r_ceil = r_floor + 1
            half_sum = 0.5 * (sorted_sample[r_floor-1] + sorted_sample[n-r_ceil])

        return half_sum


class Mean(Estimation):
    """Оценка выборочной средней."""
    def estimate(self, sample):
        """
        Оценить параметр масштаба с помощью среднего.

        Args:
            sample: Выборка, используемая для оценки.

        Returns:
            Значения расчетного параметра масштаба.
        """
        return statistics.mean(sample)


class Var(Estimation):
    """Оценка выборочной дисперсии."""
    def estimate(self, sample):
        """
        Оценка параметра сдвига с помощью оценки дисперсии.

        Args:
             sample: Выборка, используемая для оценки.

        Returns:
            Расчет масштаба сдвига.
        """
        return statistics.variance(sample)


class Modelling(ABC):
    """Абстрактный базовый класс для моделирования."""
    def __init__(self, gen: RandomNumberGenerator, estimations: list, M: int, truth_value: float):
        """
        Инициализация моделирования с указанными параметрами.

        Args:
            gen: Генератор случайных чисел.
            estimations (list) : Список оценок, которые необходимо выполнить.
            M (int) : Количество ре-выборок.
            truth_value (float) : Истинное значение.
        """
        self.gen = gen
        self.estimations = estimations
        self.M = M
        self.truth_value = truth_value

        self.estimations_sample = np.zeros((self.M, len(self.estimations)), dtype=np.float64)

    def estimate_bias_sqr(self):
        """
        Оценить квадрат смещения оценок.

        Returns:
            Массив оценочных квадратичных значений смещения.
        """
        return np.array([(Mean().estimate(self.estimations_sample[:, i]) - self.truth_value) ** 2 for i in
                         range(len(self.estimations))])

    def estimate_var(self):
        """
        Оценить дисперсию оценок.

        Returns:
            Массив оценочных значений дисперсии.
        """
        return np.array([Var().estimate(self.estimations_sample[:, i]) for i in range(len(self.estimations))])

    def estimate_mse(self):
        """
        Оценить среднеквартичную ошибку оценок.

        Returns:
            Массив оценочных значений среднеквадратичной оценки.
        """
        return self.estimate_bias_sqr() + self.estimate_var()

    def get_samples(self):
        """
        Получить оценочные образцы.

        Returns:
            Выборка в виде двумерного массива.
        """
        return self.estimations_sample

    def get_sample(self, N):
        """
        Получить сгенерированную выборку.

        Returns:
            Сгенерированная выборка.
        """
        return self.gen.get(N)

    def run(self):
        """
        Запуск моделирования оценки сдвига.

        Выполнить указанное количество итераций с сохранением образца оценки.
        """
        for i in range(self.M):
            sample = self.get_sample(N)
            self.estimations_sample[i, :] = [e.estimate(sample) for e in self.estimations]


class SmoothedRandomVariable(RandomVariable):
    """Класс, представляющий сглаженную случайную величину."""
    @staticmethod
    def _k(x):
        """
        Внутренний метод. Вычисляет функцию ядра для сглаживания.

        Args:
            x (float) : Значение.

        Returns:
            Значение функции ядра для задаенного значения.
        """
        if abs(x) <= 1:
            return 0.75 * (1 - x * x)
        else:
            return 0

    @staticmethod
    def _K(x):
        """
        Внутренний метод. Вычисляет функцию ядяра для сглаживания.

        Args:
            x (float) : Значение.

        Returns:
            Значение функции ядра для заданного значения.
        """
        if x < -1:
            return 0
        elif -1 <= x < 1:
            return 0.5 + 0.75 * (x - x ** 3 / 3)
        else:
            return 1

    def __init__(self, sample, h):
        """
        Инициализация параметров для сглаживания случайной величины.

        Args:
            sample: Выборка.
            h: Параметр сглаживания.
        """
        self.sample = sample
        self.h = h

    def pdf(self, x):
        """
        Вычисляет функцию плотности вероятности.

        Args:
            x: Значение, для которого нужно вычислить функцию плотности вероятности.

        Returns:
            Значение плотности вероятности для случайной величины.
        """
        return np.mean([SmoothedRandomVariable._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        """
        Вычисляет функцию распределения.

        Args:
            x: Значение, для которого нужно вычислить функцию распределения.

        Returns:
            Значение функции распределения для данной случайной величины.
        """
        return np.mean([SmoothedRandomVariable._K((x - y) / self.h) for y in self.sample])

    def quantile(self, alpha):
        """
        Вычисляет квантиль для данной случайной величины.

        Args:
            alpha: Уровень значимости.

        Raises:
            NotImplementedError: Метод не реализован для данной случайной величины.
        """
        raise NotImplementedError


location = 0
scale = 1
N = 500
resample_count = 100
bandwidth = 0.05

rv = LaplaceDistribution(location, scale)
generator = SimpleRandomNumberGenerator(rv)
sample = generator.get(N)
rv1 = NonParametricRandomVariable(sample)
generator1 = SimpleRandomNumberGenerator(rv1)
generator2 = TukeyRandomNumberGenerator(sample, location, 2, 0.1)

modelling = Modelling(generator2, [HodgesLehmannEstimation(), HalfSumEstimation()], resample_count,
                      location)
modelling.run()
estimate_mse = modelling.estimate_mse()
print(estimate_mse)
print(f'Первая оценка / Вторая оценка: {estimate_mse[0] / estimate_mse[1]}')
print(f'Вторая оценка / Первая оценка: {estimate_mse[1] / estimate_mse[0]}')

samples = modelling.get_samples()
POINTS = 100

for i in range(samples.shape[1]):
    sample = samples[:, i]
    X_min = min(sample)
    X_max = max(sample)
    x = np.linspace(X_min, X_max, POINTS)
    srv = SmoothedRandomVariable(sample, bandwidth)
    y = np.vectorize(srv.pdf)(x)
    plt.plot(x, y)
plt.show()
