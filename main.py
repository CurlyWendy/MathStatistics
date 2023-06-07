from abc import ABC, abstractmethod
import math
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt


class RandomVariable(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def cdf(self, x):
        pass

    @abstractmethod
    def quantile(self, alpha):
        pass


class LaplaceDistribution(RandomVariable):
    def __init__(self, location, scale):
        self.location = location
        self.scale = scale

    def pdf(self, x):
        return (1 / (2 * self.scale)) * np.exp(-np.abs(x - self.location) / self.scale)

    def cdf(self, x):
        if x < self.location:
            return 0.5 * np.exp((x - self.location) / self.scale)
        else:
            return 1 - 0.5 * np.exp(-(x - self.location) / self.scale)

    def quantile(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("Численное значение квантиля может быть в промежутке между 0 и 1.")
        if alpha == 0.5:
            return self.location
        elif alpha < 0.5:
            return self.location + self.scale * np.log(2 * alpha)
        else:
            return self.location - self.scale * np.log(2 * (1 - alpha))


class NonParametricRandomVariable(RandomVariable):
    def __init__(self, source_sample) -> None:
        super().__init__()
        self.source_sample = sorted(source_sample)

    def pdf(self, x):
        if x in self.source_sample:
            return float('inf')
        return 0

    @staticmethod
    def heaviside_function(x):
        if x > 0:
            return 1
        else:
            return 0

    def cdf(self, x):
        return np.mean(np.vectorize(self.heaviside_function)(x - self.source_sample))

    def quantile(self, alpha):
        index = int(alpha * len(self.source_sample))
        return self.source_sample[index]


class RandomNumberGenerator(ABC):
    def __init__(self, random_variable: RandomVariable):
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        pass


class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable):
        super().__init__(random_variable)

    def get(self, N):
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)


class TukeyRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, sample, location, scale, proc):
        self.data = sample
        self.location = location
        self.scale = scale
        self.proc = proc

    def get(self, N):
        proc_count = int(self.proc * N)
        outliersRV = LaplaceDistribution(self.location, self.scale)
        generator = SimpleRandomNumberGenerator(outliersRV)
        outliers = generator.get(proc_count)
        data_proc = np.concatenate((self.data, outliers))
        return data_proc


class Estimation(ABC):
    @abstractmethod
    def estimate(self, sample):
        pass


class HodgesLehmannEstimation(Estimation):
    def estimate(self, sample):
        n = len(sample)
        means = [(sample[i] + sample[j]) / 2 for i in range(n) for j in range(i, n)]
        return np.median(means)


class HalfSumEstimation(Estimation):
    def estimate(self, sample):
        sorted_sample = sorted(sample)
        n = len(sorted_sample)
        r = 0.5 * n  # выбираем полусумму порядковых статистик


        if r.is_integer():
            r = int(r)
            half_sum = 0.5 * (sorted_sample[r-1] + sorted_sample[n-r])  # полусумма двух соседних статистик
        else:
            r_floor = int(r // 1)  # нижняя целая часть
            r_ceil = r_floor + 1  # верхняя целая часть
            half_sum = 0.5 * (sorted_sample[r_floor-1] + sorted_sample[n-r_ceil])  # полусумма двух соседних статистик

        return half_sum


class Mean(Estimation):
    def estimate(self, sample):
        return statistics.mean(sample)



class Var(Estimation):
    def estimate(self, sample):
        return statistics.variance(sample)


class Modelling(ABC):
    def __init__(self, gen: RandomNumberGenerator, estimations: list, M: int, truth_value: float):
        self.gen = gen
        self.estimations = estimations
        self.M = M
        self.truth_value = truth_value

        # Здесь будут храниться выборки оценок
        self.estimations_sample = np.zeros((self.M, len(self.estimations)), dtype=np.float64)

    # Метод, оценивающий квадрат смещения оценок
    def estimate_bias_sqr(self):
        return np.array([(Mean().estimate(self.estimations_sample[:, i]) - self.truth_value) ** 2 for i in
                         range(len(self.estimations))])

    # Метод, оценивающий дисперсию оценок
    def estimate_var(self):
        return np.array([Var().estimate(self.estimations_sample[:, i]) for i in range(len(self.estimations))])

    # Метод, оценивающий СКО оценок
    def estimate_mse(self):
        return self.estimate_bias_sqr() + self.estimate_var()

    def get_samples(self):
        return self.estimations_sample

    def get_sample(self, N):
        return self.gen.get(N)

    def run(self):
        for i in range(self.M):
            sample = self.get_sample(N)
            self.estimations_sample[i, :] = [e.estimate(sample) for e in self.estimations]


class SmoothedRandomVariable(RandomVariable):
    @staticmethod
    def _k(x):
        if abs(x) <= 1:
            return 0.75 * (1 - x * x)
        else:
            return 0

    @staticmethod
    def _K(x):
        if x < -1:
            return 0
        elif -1 <= x < 1:
            return 0.5 + 0.75 * (x - x ** 3 / 3)
        else:
            return 1

    def __init__(self, sample, h):
        self.sample = sample
        self.h = h

    def pdf(self, x):
        return np.mean([SmoothedRandomVariable._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        return np.mean([SmoothedRandomVariable._K((x - y) / self.h) for y in self.sample])

    def quantile(self, alpha):
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
