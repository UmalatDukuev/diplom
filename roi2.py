import numpy as np
import matplotlib.pyplot as plt


# Определение центров, к которым стремятся частицы
def function1(x):
    return x ** 2 - 5

def function2(x):
    return np.sin(x) + 5


# Определение центров, к которым стремятся частицы (экстремумы функций)
centers = np.array([[0, function1(0)], [np.pi, function2(np.pi)]])
1

# Функция для вычисления приспособленности (fitness) каждой частицы
def calculate_fitness(particles):
    # Инициализация массива для приспособленности
    fitness = np.zeros((len(particles), len(centers)))

    # Вычисление приспособленности относительно каждого центра
    for i, center in enumerate(centers):
        fitness[:, i] = 1 / (1 + np.sqrt((particles[:, 0] - center[0]) ** 2 + (particles[:, 1] - center[1]) ** 2))

    # Возвращаем максимальное значение приспособленности для каждой частицы
    return np.max(fitness, axis=1)


# Функция для обновления скорости частиц
def update_velocity(particles, velocities, personal_best_positions, global_best_position, inertia_weight,
                    cognitive_weight, social_weight):
    # Генерация случайных чисел для обновления скорости
    r1 = np.random.rand(len(particles), 2)
    r2 = np.random.rand(len(particles), 2)

    # Обновление скорости
    new_velocities = inertia_weight * velocities + \
                     cognitive_weight * r1 * (personal_best_positions - particles) + \
                     social_weight * r2 * (global_best_position - particles)

    return new_velocities


# Функция для обновления позиций частиц
def update_position(particles, velocities):
    # Обновление позиций
    new_particles = particles + velocities
    return new_particles


# Функция для визуализации текущего состояния частиц
def plot_particles(particles, personal_best_positions, global_best_position, generation):
    plt.clf()  # Очистка предыдущего графика

    plt.scatter(particles[:, 0], particles[:, 1], color='blue', label='Particles')
    plt.scatter(personal_best_positions[:, 0], personal_best_positions[:, 1], color='green', label='Personal Best')
    plt.scatter(global_best_position[0], global_best_position[1], color='red', label='Global Best')

    # Отображение локальных центров
    plt.scatter(centers[:, 0], centers[:, 1], color='black', s=200, marker='o')

    plt.title(f'Current Particles (Generation: {generation})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.pause(0.7)  # Остановка на более длительное время для обновления графика


# Основная функция алгоритма роя частиц
def particle_swarm_optimization(num_particles, num_generations, inertia_weight=0.5, cognitive_weight=1,
                                social_weight=1):
    # Инициализация позиций и скоростей частиц
    particles = np.random.uniform(low=-10, high=10, size=(num_particles, 2))
    velocities = np.random.uniform(low=-1, high=1, size=(num_particles, 2))

    personal_best_positions = particles.copy()
    personal_best_fitness = calculate_fitness(particles)
    global_best_index = np.argmax(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index]

    for generation in range(num_generations):
        fitness = calculate_fitness(particles)

        # Обновление лучших позиций каждой частицы
        update_indices = fitness > personal_best_fitness
        personal_best_positions[update_indices] = particles[update_indices]
        personal_best_fitness[update_indices] = fitness[update_indices]

        # Обновление глобальной лучшей позиции
        global_best_index = np.argmax(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]

        # Обновление скоростей и позиций частиц
        velocities = update_velocity(particles, velocities, personal_best_positions, global_best_position,
                                     inertia_weight, cognitive_weight, social_weight)
        particles = update_position(particles, velocities)

        # Визуализация текущего состояния частиц
        plot_particles(particles, personal_best_positions, global_best_position, generation)


# Параметры алгоритма роя частиц
num_particles = 50
num_generations = 100

# Запуск алгоритма роя частиц
plt.ion()  # Включение интерактивного режима построения графиков
particle_swarm_optimization(num_particles, num_generations)
plt.ioff()  # Выключение интерактивного режима после завершения алгоритма
plt.show()  # Показать окно графика
