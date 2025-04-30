import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

#Data Acquisition
## Set the directory
PROJECT_PATH = "C:/Users\shams/PhD Implementation testing/MRFO_Levy Flight"

# Configurations
DATASET_PATH = "dataset"
CLASSES = ["malignant", "benign"]
NEW_SIZE = (128, 128, 3)
TRAIN_SIZE = 0.8
POPULATION_SIZE = 10
NO_OF_ITERATIONS = 20
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
EPOCHS = 1000
PATIENCE = 10

## Helper Functions
import matplotlib.pyplot as plt

def DisplayColorImage(image):
  newImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  f = plt.figure()
  plt.imshow(newImage)
  plt.show()
  plt.close()

def PlotHistory(history):
  f = plt.figure()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title("Accuracy Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend(["Train", "Validation"])
  plt.grid("both")
  plt.show()
  plt.close()
  
 ## Reading Dataset
 import cv2, os
X = []
y = []
counter = {"malignant": 0, "benign": 0}

for cls in CLASSES:
  clsContent = os.listdir(os.path.join(DATASET_PATH, cls))
  for imgName in clsContent:
    imgPath = os.path.join(DATASET_PATH, cls, imgName)
    image = cv2.imread(imgPath)
    image = cv2.resize(image, NEW_SIZE[:2], interpolation=cv2.INTER_CUBIC)
    X.append(image)
    y.append(cls)
    counter[cls] += 1

print(len(X), len(y))
print(counter)

DisplayColorImage(X[0])

## Dataset Balancing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataGen = ImageDataGenerator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  zoom_range=0.2,
  shear_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
)

malignantX = [X[i] for i in range(len(X)) if y[i] == "malignant"]
dataGen.fit(malignantX)

import numpy as np
diff = counter["malignant"] - counter["benign"]
imgGenerator = dataGen.flow(np.array(malignantX), batch_size=1)
newImage = next(imgGenerator)[0].astype("uint8")
print(newImage.shape)
DisplayColorImage(newImage)
# print(newImage)

# BLOCK 4
desired_dataset_size = 5000  # Set the desired dataset size to xxxxx
for i in range(desired_dataset_size - len(X)):  # Calculate how many additional samples are needed
    newImage = next(imgGenerator)[0].astype("uint8")
    X.append(newImage)
    y.append("malignant")
    counter["malignant"] += 1

print(counter)

## Data Scaling
X = np.array(X) / 255.0 # Normalization

## Label Encoding
from sklearn.preprocessing import LabelEncoder
objEnc = LabelEncoder()
yEnc = objEnc.fit_transform(y)
print(y[0], yEnc[0])

## Dataset Split
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(np.array(X), yEnc, train_size=TRAIN_SIZE, stratify=yEnc)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, train_size=TRAIN_SIZE, stratify=trainY)
print(trainX.shape)
print(testX.shape)
print(valX.shape)

## Learning and Optimization
#from tensorflow.keras.optimizers.legacy import *
from tensorflow.keras.optimizers import *

ranges = {
  "Rotation": np.arange(0, 11, 1),
  "Width Shift": np.arange(0, 0.11, 0.05),
  "Height Shift": np.arange(0, 0.11, 0.05),
  "Zoom": np.arange(0, 0.11, 0.05),
  "Shear": np.arange(0, 0.11, 0.05),
  "Horizontal Flip": [True, False],
  "Vertical Flip": [True, False],
  "Optimizer": [Adam(), Nadam(), RMSprop(), Adadelta(), Adagrad(), SGD()],
  "Batch Size": [8, 16, 32, 64],
  "TL Learn Ratio": np.arange(0, 26, 1),
}

SOLUTION_SIZE = len(ranges.keys())
SOLUTION_SIZE

# Population Initialization
population = np.random.uniform(
  low=LOWER_BOUND,
  high=UPPER_BOUND,
  size=(POPULATION_SIZE, SOLUTION_SIZE)
)
print(population.shape)
print(population[0])

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np



from tensorflow.keras.models import Model
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input

# Fitness Function Evaluation
def FitnessFunction(solution):
  solution = np.round(solution, 4)

  index = int(np.round(solution[0] * (len(ranges["Rotation"]) - 1)))
  rotationValue = ranges["Rotation"][index]

  index = int(np.round(solution[1] * (len(ranges["Width Shift"]) - 1)))
  widthShiftValue = ranges["Width Shift"][index]

  index = int(np.round(solution[2] * (len(ranges["Height Shift"]) - 1)))
  heightShiftValue = ranges["Height Shift"][index]

  index = int(np.round(solution[3] * (len(ranges["Zoom"]) - 1)))
  zoomValue = ranges["Zoom"][index]

  index = int(np.round(solution[4] * (len(ranges["Shear"]) - 1)))
  shearValue = ranges["Shear"][index]

  index = int(np.round(solution[5] * (len(ranges["Horizontal Flip"]) - 1)))
  hFlipValue = ranges["Horizontal Flip"][index]

  index = int(np.round(solution[6] * (len(ranges["Vertical Flip"]) - 1)))
  vFlipValue = ranges["Vertical Flip"][index]

  index = int(np.round(solution[7] * (len(ranges["Optimizer"]) - 1)))
  optimizerValue = ranges["Optimizer"][index]

  index = int(np.round(solution[8] * (len(ranges["Batch Size"]) - 1)))
  batchSizeValue = ranges["Batch Size"][index]

  index = int(np.round(solution[9] * (len(ranges["TL Learn Ratio"]) - 1)))
  tlLearnRatioValue = ranges["TL Learn Ratio"][index]

  dataGen = ImageDataGenerator(
    rotation_range=rotationValue,
    width_shift_range=widthShiftValue,
    height_shift_range=heightShiftValue,
    zoom_range=zoomValue,
    shear_range=shearValue,
    horizontal_flip=hFlipValue,
    vertical_flip=vFlipValue,
  )

  baseModel = VGG19(weights='imagenet', include_top=False, input_tensor=Input(NEW_SIZE))

  x = baseModel.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(1, activation='sigmoid')(x)

  model = Model(inputs=baseModel.input, outputs=predictions)

  for layer in baseModel.layers:
    layer.trainable = False

  fromIndex = int(np.round((len(baseModel.layers) - 1) * (1.0 - tlLearnRatioValue / 100.0)))
  for layer in baseModel.layers[fromIndex:]:
    layer.trainable = True

  model.compile(
    optimizer=optimizerValue,
    loss='binary_crossentropy',
    metrics=["accuracy", Precision(), Recall(), AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()]
  )

  keyword = "VGG19_" + "-".join([str(el)[2:] for el in solution])
  checkpointPath = os.path.join(PROJECT_PATH, "Checkpoints", keyword) + ".h5"
  csvLogPath = os.path.join(PROJECT_PATH, "Logs", keyword) + ".csv"

  histroy = model.fit(
    dataGen.flow(trainX, trainY, batch_size=batchSizeValue),
    validation_data=(valX, valY),
    batch_size=batchSizeValue,
    epochs=EPOCHS,
    verbose=0,
    callbacks=[
      ModelCheckpoint(checkpointPath, save_best_only=True, save_weights_only=True, monitor="val_accuracy", mode="max", verbose=0),
      TerminateOnNaN(),
      CSVLogger(csvLogPath, append=True),
      EarlyStopping(monitor="val_accuracy", mode="max", patience=PATIENCE),
    ]
  )

  #PlotHistory(histroy)

  model.load_weights(checkpointPath)
  scoresList = model.evaluate(testX, testY, verbose=0)
  score = (scoresList[1] + scoresList[2] + scoresList[3]) / 3.0

  configs = [
    rotationValue,
    widthShiftValue,
    heightShiftValue,
    zoomValue,
    shearValue,
    hFlipValue,
    vFlipValue,
    optimizerValue._name,
    batchSizeValue,
    tlLearnRatioValue,
  ]
  print(scoresList, score, configs)

  return score
  

from scipy.stats import levy

def levy_flight(beta, solution_size):
    numSteps = np.random.randint(1, solution_size+1)
    xLevy = 0.01 * levy.rvs(beta, size=numSteps)
    xLevy = xLevy / np.sum(np.abs(xLevy))
    return xLevy

def PopulationUpdating(population, scores, iterationNumber):
    bestIndex = np.argmax(scores)
    bestSolution = population[bestIndex].copy()
    bestScore = scores[bestIndex]

    newPopulation = population.copy()
    coef = iterationNumber / float(NO_OF_ITERATIONS)
    for i in range(len(population)):
        r = np.random.random(1)
        alpha = 2.0 * r * np.sqrt(np.abs(np.log(r)))
        r1 = np.random.random(1)
        factor = (NO_OF_ITERATIONS - iterationNumber + 1.0) / (NO_OF_ITERATIONS * 1.0)
        beta = 2.0 * np.exp(r1 * factor) * np.sin(2.0 * np.pi * r1)

        if (np.random.random(1) < 0.5):
            if (coef < np.random.random(1)):
                s = np.subtract(UPPER_BOUND, LOWER_BOUND)
                u = np.random.uniform(low=0, high=1, size=SOLUTION_SIZE)
                m = np.multiply(u, s)
                xRand = np.clip(np.add(LOWER_BOUND, m), LOWER_BOUND, UPPER_BOUND)
                if (i == 0):
                    levy_step = np.array([levy_flight(beta=1.5, solution_size=1) for _ in range(SOLUTION_SIZE)]).flatten()
                    newPopulation[i, :] = xRand + r * (xRand - population[i, :]) + beta * (xRand - population[i, :]) + levy_step
                else:
                    levy_step = np.array([levy_flight(beta=1.5, solution_size=1) for _ in range(SOLUTION_SIZE)]).flatten()
                    newPopulation[i, :] = xRand + r * (bestSolution - population[i, :]) + beta * (xRand - population[i, :]) + levy_step
            else:
                if (i == 0):
                    levy_step = np.array([levy_flight(beta=1.5, solution_size=1) for _ in range(SOLUTION_SIZE)]).flatten()
                    newPopulation[i, :] = bestSolution + r * (bestSolution - population[i, :]) + beta * (bestSolution - population[i, :]) + levy_step
                else:
                    levy_step = np.array([levy_flight(beta=1.5, solution_size=1) for _ in range(SOLUTION_SIZE)]).flatten()
                    newPopulation[i, :] = bestSolution + r * (bestSolution - population[i, :]) + beta * (bestSolution - population[i, :]) + levy_step
        else:
            if bestSolution.shape == population[i, :].shape:
                if (i == 0):
                    levy_step = np.array([levy_flight(beta=1.5, solution_size=1) for _ in range(SOLUTION_SIZE)]).flatten()
                    newPopulation[i, :] = population[i, :] + r * (bestSolution - population[i, :]) + alpha * (bestSolution - population[i, :]) + levy_step
                else:
                    levy_step = np.array([levy_flight(beta=1.5, solution_size=1) for _ in range(SOLUTION_SIZE)]).flatten()
                    newPopulation[i, :] = population[i, :] + r * (bestSolution - population[i, :]) + alpha * (bestSolution - population[i, :]) + levy_step
            else:
                print(f"Warning: bestSolution shape {bestSolution.shape} is not compatible with population shape {population[i, :].shape}. Skipping this step.")

        newPopulation[i, :] = np.clip(newPopulation[i, :], LOWER_BOUND, UPPER_BOUND)
        currentScore = FitnessFunction(newPopulation[i, :])
        if (currentScore > bestScore):
            bestSolution, bestScore = newPopulation[i, :].copy(), currentScore

        s = 2.0
        r2, r3 = np.random.random(1), np.random.random(1)
        newPopulation[i, :] = population[i, :] + s * (r2 * bestSolution - r3 * population[i, :])
        newPopulation[i, :] = np.clip(newPopulation[i, :], LOWER_BOUND, UPPER_BOUND)
        currentScore = FitnessFunction(newPopulation[i, :])
        if (currentScore > bestScore):
            bestSolution, bestScore = newPopulation[i, :].copy(), currentScore

    return newPopulation.copy()


import numpy as np

# Initialize lists to store data
best_scores_with_levy = []
mean_scores_with_levy = []
diversity_with_levy = []
NUM_INDEPENDENT_RUNS = 20
final_scores_with_levy = []
import time
def calculate_diversity(population):
    diversity = np.std(population, axis=0)
    return np.mean(diversity)
start_time = time.time()
for run in range(NUM_INDEPENDENT_RUNS):
    bestSolutions = []
    bestScores = []
    mean_scores_run = []  # List to store mean scores for each run

    for iterationNumber in range(NO_OF_ITERATIONS):
        scores = []
        for i in range(len(population)):
            score = FitnessFunction(population[i])
            scores.append(score)
        newPopulation = PopulationUpdating(population, scores, iterationNumber)

        bestIndex = np.argmax(scores)
        bestSolution = population[bestIndex].copy()
        bestScore = scores[bestIndex]

        bestSolutions.append(bestSolution)
        bestScores.append(bestScore)

        # Append data for analysis
        mean_score_with_levy = np.mean(scores)
        mean_scores_run.append(mean_score_with_levy)  # Append to the run-specific list
        diversity_with_levy.append(calculate_diversity(population))

        population = newPopulation.copy()

    # Append the best score of each independent run
    best_scores_with_levy.append(bestScores[-1])

    # Append the final best fitness score to the respective list
    final_scores_with_levy.append(bestScores[-1])

    # Concatenate the mean scores for this run to the overall list
    mean_scores_with_levy.extend(mean_scores_run)
end_time = time.time()  # ⏱️ Stop timer after all runs
total_minutes = (end_time - start_time) / 60
print(f"Total Runtime for {NUM_INDEPENDENT_RUNS} runs: {total_minutes:.2f} minutes")

 import os

# LOGGING THE DATA
population_scores_path = os.path.join(PROJECT_PATH, "Population.csv")

with open(population_scores_path, "a") as file:
    for i, sol in enumerate(population):
        score = scores[i]
        data = f"{iterationNumber + 1},{i + 1}," + ",".join(str(el) for el in sol) + f",{score}\n"
        file.write(data)

best_solutions_path = os.path.join(PROJECT_PATH, "BestSolutions.csv")

with open(best_solutions_path, "w") as file:
    for i, sol in enumerate(bestSolutions):
        score = bestScores[i]
        data = ",".join(str(el) for el in sol) + f",{score}\n"
        file.write(data)

import matplotlib.pyplot as plt

# Best Fitness Score Comparison
plt.figure(figsize=(10, 6))
# Ensure the data is not empty
if len(best_scores_with_levy) > 0:
    plt.plot(range(1, len(best_scores_with_levy) + 1), best_scores_with_levy, label='With Levy Flights')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Score')
    plt.title('Best Fitness Score Comparison')
    plt.legend()
    plt.show()
else:
    print("No data available for best_scores_with_levy.")

# Convergence Behavior
for run in range(NUM_INDEPENDENT_RUNS):
    plt.figure(figsize=(10, 6))
    start_idx = run * NO_OF_ITERATIONS
    end_idx = (run + 1) * NO_OF_ITERATIONS

    # Add check before plotting
    if len(mean_scores_with_levy[start_idx:end_idx]) == 0:
        print(f"Skipping Run {run+1}: No data available.")
        continue

    plt.plot(range(1, NO_OF_ITERATIONS+1), mean_scores_with_levy[start_idx:end_idx], label=f'Run {run+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Fitness Score')
    plt.title(f'Convergence Behavior (Run {run+1})')
    plt.legend()
    plt.show()

# Exploration Capabilities
for run in range(NUM_INDEPENDENT_RUNS):
    plt.figure(figsize=(10, 6))
    start_idx = run * NO_OF_ITERATIONS
    end_idx = (run + 1) * NO_OF_ITERATIONS

    # Add check before plotting
    if len(diversity_with_levy[start_idx:end_idx]) == 0:
        print(f"Skipping Run {run+1}: No data available.")
        continue

    plt.plot(range(1, NO_OF_ITERATIONS+1), diversity_with_levy[start_idx:end_idx], label=f'Run {run+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Population Diversity')
    plt.title(f'Exploration Capabilities (Run {run+1})')
    plt.legend()
    plt.show()

# Stability and Robustness
plt.figure(figsize=(10, 6))
# Ensure the data is not empty
if len(final_scores_with_levy) > 0:
    plt.boxplot(final_scores_with_levy, labels=['With Levy Flights'])
    plt.title('Final Best Fitness Scores')
    plt.show()
else:
    print("No data available for final_scores_with_levy.")

# Population Diversity over iterations
for run in range(NUM_INDEPENDENT_RUNS):
    start_idx = run * NO_OF_ITERATIONS
    end_idx = (run + 1) * NO_OF_ITERATIONS

    # Add check before plotting
    if len(diversity_with_levy[start_idx:end_idx]) == 0:
        print(f"Skipping Run {run+1}: No data available.")
        continue

    plt.plot(range(1, NO_OF_ITERATIONS+1), diversity_with_levy[start_idx:end_idx], label=f'Run {run+1}')
    
plt.xlabel('Iteration')
plt.ylabel('Population Diversity')
plt.title('Population Diversity Analysis')
plt.legend()
plt.show()

# Algorithm dynamics (3D plot)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ensure population data is not empty
if len(population) > 0:
    for run in range(NUM_INDEPENDENT_RUNS):
        for iteration in range(NO_OF_ITERATIONS):
            start_idx = run * NO_OF_ITERATIONS + iteration
            end_idx = start_idx + len(population)
            
            # Ensure the slice is not empty before plotting
            if len(population[start_idx:end_idx]) == 0:
                print(f"Skipping Run {run+1}, Iteration {iteration+1}: No data available.")
                continue

            individuals = population[start_idx:end_idx]
            ax.scatter(individuals[:, 0], individuals[:, 1], iteration, marker='o', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Iteration')
    plt.title('Algorithm Dynamics')
    plt.show()
else:
    print("No data available for population.")


# 1. Convergence Analysis
def analyze_convergence(best_scores_with_levy, mean_scores_with_levy, NUM_INDEPENDENT_RUNS, NO_OF_ITERATIONS, convergence_threshold=1e-6):
    # Plot the best fitness score achieved at each iteration across multiple independent runs
    plt.figure(figsize=(10, 6))
    for run in range(NUM_INDEPENDENT_RUNS):
        plt.plot(range(1, NO_OF_ITERATIONS+1), best_scores_with_levy[run], label=f'Run {run+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Score')
    plt.title('Convergence Behavior')
    plt.legend()
    plt.show()

    # Calculate the average number of iterations required for the algorithm to converge
    convergence_iterations = []
    for run in range(NUM_INDEPENDENT_RUNS):
        best_scores = best_scores_with_levy[run]
        converged = False
        for iteration in range(1, NO_OF_ITERATIONS):
            if abs(best_scores[iteration] - best_scores[iteration-1]) < convergence_threshold:
                convergence_iterations.append(iteration)
                converged = True
                break
        if not converged:
            convergence_iterations.append(NO_OF_ITERATIONS)

    avg_convergence_iterations = np.mean(convergence_iterations)
    print(f"Average number of iterations required for convergence: {avg_convergence_iterations:.2f}")

# 2. Solution Quality Analysis
def analyze_solution_quality(final_scores_with_levy, optimal_solution=None):
    # Compare the final best fitness scores achieved by the algorithm with known optimal solutions
    if optimal_solution is not None:
        print(f"Known optimal solution: {optimal_solution}")
    print(f"Final best fitness scores achieved: {final_scores_with_levy}")

    # Analyze the distribution of final best fitness scores across independent runs
    plt.figure(figsize=(10, 6))
    plt.boxplot(final_scores_with_levy, labels=['With Levy Flights'])
    plt.title('Final Best Fitness Scores')
    plt.show()


from mpl_toolkits.mplot3d import Axes3D
#Algorithm dynamics
# Visualize the movement of individuals in the search space (assuming a 2D search space)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for run in range(NUM_INDEPENDENT_RUNS):
    for iteration in range(NO_OF_ITERATIONS):
        start_idx = run * NO_OF_ITERATIONS + iteration
        end_idx = start_idx + len(population)
        individuals = population[start_idx:end_idx]
        ax.scatter(individuals[:, 0], individuals[:, 1], iteration, marker='o', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Iteration')
plt.title('Algorithm Dynamics')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Best Fitness Score Comparison
plt.figure(figsize=(10, 6))
# Ensure the data is not empty
if len(best_scores_with_levy) > 0:
    plt.plot(range(1, len(best_scores_with_levy) + 1), best_scores_with_levy, label='With Levy Flights')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Score')
    plt.title('Best Fitness Score Comparison')
    plt.legend()
    plt.show()
else:
    print("No data available for best_scores_with_levy.")

# Convergence Behavior
for run in range(NUM_INDEPENDENT_RUNS):
    plt.figure(figsize=(10, 6))
    start_idx = run * NO_OF_ITERATIONS
    end_idx = (run + 1) * NO_OF_ITERATIONS

    # Add check before plotting
    if len(mean_scores_with_levy[start_idx:end_idx]) == NO_OF_ITERATIONS:
        plt.plot(range(1, NO_OF_ITERATIONS + 1), mean_scores_with_levy[start_idx:end_idx], label=f'Run {run+1}')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Fitness Score')
        plt.title(f'Convergence Behavior (Run {run+1})')
        plt.legend()
        plt.show()
    else:
        print(f"Skipping Run {run+1}: Mismatched lengths for x and y.")

# Exploration Capabilities
for run in range(NUM_INDEPENDENT_RUNS):
    plt.figure(figsize=(10, 6))
    start_idx = run * NO_OF_ITERATIONS
    end_idx = (run + 1) * NO_OF_ITERATIONS

    # Add check before plotting
    if len(diversity_with_levy[start_idx:end_idx]) == NO_OF_ITERATIONS:
        plt.plot(range(1, NO_OF_ITERATIONS + 1), diversity_with_levy[start_idx:end_idx], label=f'Run {run+1}')
        plt.xlabel('Iteration')
        plt.ylabel('Population Diversity')
        plt.title(f'Exploration Capabilities (Run {run+1})')
        plt.legend()
        plt.show()
    else:
        print(f"Skipping Run {run+1}: Mismatched lengths for x and y.")

# Stability and Robustness
plt.figure(figsize=(10, 6))
# Ensure the data is not empty
if len(final_scores_with_levy) > 0:
    plt.boxplot(final_scores_with_levy, labels=['With Levy Flights'])
    plt.title('Final Best Fitness Scores')
    plt.show()
else:
    print("No data available for final_scores_with_levy.")

# Population Diversity over iterations
for run in range(NUM_INDEPENDENT_RUNS):
    start_idx = run * NO_OF_ITERATIONS
    end_idx = (run + 1) * NO_OF_ITERATIONS

    # Add check before plotting
    if len(diversity_with_levy[start_idx:end_idx]) == NO_OF_ITERATIONS:
        plt.plot(range(1, NO_OF_ITERATIONS + 1), diversity_with_levy[start_idx:end_idx], label=f'Run {run+1}')
    else:
        print(f"Skipping Run {run+1}: Mismatched lengths for x and y.")

plt.xlabel('Iteration')
plt.ylabel('Population Diversity')
plt.title('Population Diversity Analysis')
plt.legend()
plt.show()

# Algorithm dynamics (3D plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ensure population data is not empty
if len(population) > 0:
    for run in range(NUM_INDEPENDENT_RUNS):
        for iteration in range(NO_OF_ITERATIONS):
            start_idx = run * NO_OF_ITERATIONS + iteration
            end_idx = start_idx + len(population)

            # Ensure the slice is not empty and has the correct dimensions before plotting
            if len(population[start_idx:end_idx]) > 0:
                individuals = population[start_idx:end_idx]
                ax.scatter(individuals[:, 0], individuals[:, 1], iteration, marker='o', alpha=0.5)
            else:
                print(f"Skipping Run {run+1}, Iteration {iteration+1}: No data available.")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Iteration')
    plt.title('Algorithm Dynamics')
    plt.show()
else:
    print("No data available for population.")


# Algorithm dynamics (3D plot with debugging)
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Ensure population data is not empty
if len(population) > 0:
    for run in range(NUM_INDEPENDENT_RUNS):
        for iteration in range(NO_OF_ITERATIONS):
            start_idx = run * NO_OF_ITERATIONS + iteration
            end_idx = start_idx + len(population[0])

            # Ensure the slice is not empty and check dimensions before plotting
            if start_idx < len(population) and len(population[start_idx:end_idx]) > 0:
                individuals = np.array(population[start_idx:end_idx])

                # Debugging print to check diversity in data
                print(f"Run {run+1}, Iteration {iteration+1} sample data:\n", individuals[:5])  # Print first 5 individuals

                # Check if individuals array is 2D (required for X and Y coordinates)
                if individuals.shape[1] >= 2:
                    # Increase scaling factor for better visualization
                    x_vals = individuals[:, 0] * 1000  
                    y_vals = individuals[:, 1] * 1000
                    z_vals = np.full(x_vals.shape, iteration)  # Z-axis is the iteration number

                    # Scatter plot each point in the current iteration
                    ax.scatter(x_vals, y_vals, z_vals, marker='o', alpha=0.5)
                else:
                    print(f"Skipping Run {run+1}, Iteration {iteration+1}: Insufficient dimensions in population data.")
            else:
                print(f"Skipping Run {run+1}, Iteration {iteration+1}: No data available.")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Iteration')
    plt.title('Algorithm Dynamics')
    plt.show()
else:
    print("No data available for population.")
