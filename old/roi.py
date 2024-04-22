import random
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, dimensions):
        self.position = [random.uniform(-5, 5) for _ in range(dimensions)]  # Начальная позиция частицы
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)]  # Начальная скорость частицы
        self.best_position = self.position[:]  # Лучшая позиция частицы
        self.best_score = float('inf')  # Лучший результат (по умолчанию считаем, что функция оценки минимизируется)


def fitness_function(x):
    # Пример функции оценки (сфера)
    return sum(xi ** 2 for xi in x)


def update_velocity(particle, global_best_position, inertia=0.5, cognitive_weight=1, social_weight=2):
    for i in range(len(particle.velocity)):
        # Вычисление новой скорости частицы
        cognitive_velocity = cognitive_weight * random.random() * (particle.best_position[i] - particle.position[i])
        social_velocity = social_weight * random.random() * (global_best_position[i] - particle.position[i])
        new_velocity = inertia * particle.velocity[i] + cognitive_velocity + social_velocity
        particle.velocity[i] = new_velocity


def update_position(particle):
    # Обновление позиции частицы на основе скорости
    for i in range(len(particle.position)):
        particle.position[i] += particle.velocity[i]


def pso(num_particles, dimensions, num_iterations):
    swarm = [Particle(dimensions) for _ in range(num_particles)]  # Инициализация частиц
    global_best_position = [float('inf')] * dimensions
    global_best_score = float('inf')

    positions_history = [[] for _ in range(num_particles)]  # Сохраняем историю положений для визуализации

    for _ in range(num_iterations):
        for idx, particle in enumerate(swarm):
            positions_history[idx].append(particle.position[:])
            score = fitness_function(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position[:]
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position[:]

        for particle in swarm:
            update_velocity(particle, global_best_position)
            update_position(particle)

    return global_best_position, global_best_score, positions_history


if __name__ == "__main__":
    dimensions = 2  # Количество измерений (параметров)
    num_particles = 5  # Количество частиц
    num_iterations = 100  # Количество итераций

    best_position, best_score, positions_history = pso(num_particles, dimensions, num_iterations)

    # Визуализация движения точек на координатной плоскости
    for positions in positions_history:
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        plt.plot(x, y, marker='o')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Movement of Particles in PSO')
    plt.grid(True)
    plt.show()

    print("Best Position:", best_position)
    print("Best Score:", best_score)
