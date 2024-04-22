import numpy as np
import matplotlib.pyplot as plt

# Определение центров, к которым стремятся точки
def function1(x):
    return x ** 2 - 5

def function2(x):
    return np.sin(x) + 5

# Определение центров, к которым стремятся точки (экстремумы функций)
centers = np.array([[0, function1(0)], [np.pi, function2(np.pi)]])

# Функция для инициализации начальной популяции точек
def initialize_population(population_size, x_range=(-10, 10), y_range=(-10, 10)):
    population = np.random.uniform(low=x_range[0], high=x_range[1], size=(population_size, 2))
    return population

# Функция для вычисления приспособленности (fitness) каждой точки в популяции
def calculate_fitness(population):
    # Инициализация массива для приспособленности
    fitness = np.zeros((len(population), len(centers)))

    # Вычисление приспособленности относительно каждого центра
    for i, center in enumerate(centers):
        fitness[:, i] = 1 / (1 + np.sqrt((population[:, 0] - center[0]) ** 2 + (population[:, 1] - center[1]) ** 2))

    # Возвращаем максимальное значение приспособленности для каждой точки
    return np.max(fitness, axis=1)


# Функция для выбора родителей для скрещивания
def select_parents(population, fitness):
    # Используем метод турнира для выбора двух родителей
    indices = np.random.choice(len(population), size=2, replace=False)
    parent1 = population[indices[0]]
    parent2 = population[indices[1]]
    return parent1, parent2


# Функция для скрещивания двух родителей
def crossover(parent1, parent2):
    # Простое смешивание генов
    return (parent1 + parent2) / 2


# Функция для мутации потомка
def mutate(child, mutation_rate=0.1):
    # Добавляем случайный шум к координатам потомка
    mutation = np.random.uniform(low=-0.5, high=0.5, size=child.shape)
    mutation *= mutation_rate
    child += mutation
    return child


# Функция для визуализации текущего состояния популяции точек
def plot_population(population, generation):
    plt.clf()  # Очистка предыдущего графика

    # Определение цветов для каждой точки в зависимости от ближайшего центра
    colors = ['red', 'blue', 'green', 'orange']  # Цвета для каждого центра
    distances = np.sqrt(
        (population[:, np.newaxis, 0] - centers[:, 0]) ** 2 + (population[:, np.newaxis, 1] - centers[:, 1]) ** 2)
    nearest_center_indices = np.argmin(distances, axis=1)

    # Отображение точек с соответствующими цветами
    for i, center in enumerate(centers):
        mask = nearest_center_indices == i
        plt.scatter(population[mask, 0], population[mask, 1], color=colors[i])

    # Отображение локальных центров
    plt.scatter(centers[:, 0], centers[:, 1], color='black', s=200, marker='o')

    plt.title(f'Current Population (Generation: {generation})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.pause(0.1)  # Остановка на короткое время для обновления графика


# Основная функция генетического алгоритма
def genetic_algorithm(population_size, num_generations):
    population = initialize_population(population_size)
    for generation in range(num_generations):
        fitness = calculate_fitness(population)
        parent1, parent2 = select_parents(population, fitness)
        child = crossover(parent1, parent2)
        child = mutate(child)
        population[np.argmin(fitness)] = child

        # Визуализация текущего состояния популяции
        plot_population(population, generation)


# Параметры генетического алгоритма
population_size = 200
num_generations = 250

# Запуск генетического алгоритма
plt.ion()  # Включение интерактивного режима построения графиков
genetic_algorithm(population_size, num_generations)
plt.ioff()  # Выключение интерактивного режима после завершения алгоритма
plt.show()  # Показать окно графика
