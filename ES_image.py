"""
CISC455/851 coding practice - Evolution Strategy with one mutation_step_size self-adaptation
Written by: Ronny Rochwerg (Head TA)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 替换为支持的后端
import matplotlib.pyplot as plt


def image_fitness(image_vector, target_image):
    """Calculate fitness for an image vector. Lower is better."""
    # If no target, we'll just maximize variance (example objective)
    if target_image is None:
        return -np.var(image_vector)  # 平方差 方差越大，图像越丰富
    else:
        # Compare to target using Mean Squared Error
        return np.mean((image_vector - np.ravel(target_image)) ** 2)


def initialize_population(mu, img_size, bounds, sigma_init):
    """Initialize population of images and mutation parameters"""
    dim = img_size[0] * img_size[1]
    population = np.random.uniform(bounds[0], bounds[1], (mu, dim)) # 生成一个mu*dim的矩阵，每个元素都是0-1之间的随机数
    sigma = np.full((mu, dim), sigma_init) # 生成一个mu*dim的矩阵，每个元素都是sigma_init
    return population, sigma


def mutate(parent, parent_sigma, dim, bounds, lr, sigma_bound):
    """Mutate image with self-adapting mutation rates"""
    child_sigma = parent_sigma * np.exp(lr * np.random.normal(0, 1, size=dim)) # return a list of size dim
    child_sigma = np.clip(child_sigma, sigma_bound, None)
    child = parent + np.random.normal(0, parent_sigma, size=dim)
    child = np.clip(child, bounds[0], bounds[1])
    return child, child_sigma


def generate_offspring(population, sigma, mu, lambd, dim, bounds, lr, sigma_bound, fitness_fn):
    """Generate new offspring population"""
    offspring = []
    for _ in range(lambd):
        parent_idx = np.random.randint(mu)  # Random parent selection， 随机从mu（30）中选择一个父代index
        child, child_sigma = mutate(population[parent_idx], sigma[parent_idx],
                                    dim, bounds, lr, sigma_bound)
        fitness = fitness_fn(child)
        offspring.append((child, child_sigma, fitness))
    return offspring


def select_survivors(offspring, mu):
    """Select top mu individuals from offspring"""
    offspring.sort(key=lambda x: x[2]) # Sort by fitness， 第三个参数(child, child_sigma, fitness)
    population = np.array([x[0] for x in offspring[:mu]]) # 选择前mu个个体
    sigma = np.array([x[1] for x in offspring[:mu]]) # 选择前mu个sigma
    best_solution, best_fitness = offspring[0][0], offspring[0][2] # 选择最好的个体和fitness
    return population, sigma, best_solution, best_fitness


class ImageOptimizer:
    def __init__(self, img_size=(24, 24), target=None):
        self.img_size = img_size
        self.dim = img_size[0] * img_size[1] # 24*24=576
        self.bounds = (0, 1)  # Pixel range [0, 1]
        self.target = target

        # Setup visualization
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img_display = self.ax.imshow(np.random.rand(*img_size),
                                          cmap='gray', vmin=0, vmax=1)
        plt.title("Evolving Image")

    def update_display(self, image_vector):
        """Update the displayed image"""
        img = image_vector.reshape(self.img_size)
        self.img_display.set_data(img)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def evolve(self, mu=30, lambd=200, sigma_init=0.01,
               sigma_bound=0.001, max_generations=100000):
        """Run the evolutionary strategy"""
        population, sigma = initialize_population(mu, self.img_size,
                                                  self.bounds, sigma_init) # 初始化种群
        best_solution, best_fitness = None, float('inf')
        lr = 1 / np.sqrt(self.dim)

        for gen in range(max_generations):
            offspring = generate_offspring(population, sigma, mu, lambd,
                                           self.dim, self.bounds, lr,
                                           sigma_bound, lambda x: image_fitness(x, self.target))
            population, sigma, new_best, new_fitness = select_survivors(offspring, mu)

            if new_fitness < best_fitness:
                best_solution, best_fitness = new_best, new_fitness
                self.update_display(best_solution)

            print(f"Gen {gen}: Best Fitness {best_fitness:.4f}")

        return best_solution.reshape(self.img_size)


# Example usage:
if __name__ == "__main__":
    # Create a target image (optional)
    target = np.zeros((24, 24)) # 初始化一个24*24的全0矩阵黑色
    target[8:16, 8:16] = 1  # White square in center 8x8 中心8*8的区域为白色
    fig, ax = plt.subplots() # 创建一个画布
    img_display = ax.imshow(target, cmap='gray', vmin=0, vmax=1) # 显示图片
    plt.title("Target Image")

    # Initialize and run optimizer
    optimizer = ImageOptimizer(img_size=(24, 24), target=target)  # 初始化一个24*24的图片


    final_image = optimizer.evolve(max_generations=10000)

    # Show final result
    plt.ioff()
    plt.imshow(final_image, cmap='gray')
    plt.title("Final Evolved Image")
    plt.show()