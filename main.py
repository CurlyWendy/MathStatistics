from abc import ABC, abstractmethod
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics

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


class NormalRandomVariable(RandomVariable):
    def __init__(self, location=0, scale=1) -> None:
        super().__init__()
        self.location = location
        self.scale = scale

    def pdf(self, x):
        z = (x - self.location) / self.scale
        return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self.scale)

    def cdf(self, x):
        z = (x - self.location) / self.scale;
        if z <= 0:
            return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

    def quantile(self, alpha):
        return self.location + 4.91 * self.scale * (math.pow(alpha, 0.14) - math.pow(1 - alpha, 0.14))


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

class Estimation(ABC):
  @abstractmethod
  def estimate(sample):
    pass


class Mean(Estimation):
    def estimate(self, sample):
        return statistics.mean(sample)


class MinMax(Estimation):
    def estimate(self, sample):
        return (max(sample) + min(sample)) / 2


class Var(Estimation):
    def estimate(self, sample):
        return statistics.variance(sample)

class Modelling(ABC):
    def __init__(self, gen: RandomNumberGenerator, estimations: list, M:int, truth_value:float):
        self.gen = gen
        self.estimations = estimations
        self.M = M
        self.truth_value = truth_value

        # Здесь будут храниться выборки оценок
        self.estimations_sample = np.zeros((self.M, len(self.estimations)), dtype=np.float64)

    # Метод, оценивающий квадрат смещения оценок
    def estimate_bias_sqr(self):
        return np.array([(Mean().estimate(self.estimations_sample[:,i]) - self.truth_value) ** 2 for i in range(len(self.estimations))])

    # Метод, оценивающий дисперсию оценок
    def estimate_var(self):
        return np.array([Var().estimate(self.estimations_sample[:,i]) for i in range(len(self.estimations))])

    # Метод, оценивающий СКО оценок
    def estimate_mse(self):
        return self.estimate_bias_sqr() + self.estimate_var()

    def get_samples(self):
        return self.estimations_sample

    def get_sample(self):
        return self.gen.get(N)

    def run(self):
        for i in range(self.M):
            sample = self.get_sample()
            self.estimations_sample[i, :] = [e.estimate(sample) for e in self.estimations]

class SmoothedRandomVariable(RandomVariable):
    def _k(x):
        if abs(x) <= 1:
            return 0.75 * (1 - x * x)
        else:
            return 0

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

location = 0 #@param {type:"number"}
scale = 1 #@param {type:"number"}

N = 500  #@param {type:"integer"}

rv = NormalRandomVariable(location, scale)
generator = SimpleRandomNumberGenerator(rv)

sample = generator.get(N)
sample

rv1 = NonParametricRandomVariable(sample)
generator1 = SimpleRandomNumberGenerator(rv1)

Mean().estimate(sample)

MinMax().estimate(sample)

modelling = Modelling(generator1, [Mean(), MinMax()], 2 * N, location)
modelling.run()

mses = modelling.estimate_mse()
mses

mses[1] / mses[0]

bandwidth = 0.05 #@param {type:"number"}

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

