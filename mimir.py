"""
Mimir v1.0

An open-source, amateur data-science library aiming to
explain data science algorithms and concepts concisely
through implementation from scratch

Requires matplotlib for visualization
Other included libraries are
included in the default installation
of Python

Minimum Version of Python: 3.4

Last edited: May 09 2020, Saturday

"""

import math
import random
from collections import Counter, defaultdict
from functools import reduce, partial
import matplotlib.pyplot as plt

"""

VECTOR class

Usage: vector = Vector(...some vector list) to initialize
where the Vector list has m x 1 dimension
Vector list should be in the form [x, y, z...]

"""


class Vector:

   # Initialization of the class
   # Accepts m x 1 vector list as an argument
   def __init__(self, vector):
       self.vect = vector

   # Addition magic method
   # Subtracts that Vector's vector from the other's
   def __add__(self, other):
       return Vector([xi + xt for xi, xt in zip(self.vect, other.vect)])

   def __iadd__(self, other):
       return Vector([xi + xt for xi, xt in zip(self.vect, other.vect)])

   # Subtraction magic method
   # Subtracts that Vector's vector from the other's
   def __sub__(self, other):
       return Vector([xi - xt for xi, xt in zip(self.vect, other.vect)])

   def __isub__(self, other):
       return Vector([xi - xt for xi, xt in zip(self.vect, other.vect)])

   # Multiplication magic method for vectors
   # detects whether the other element being multipled
   # is a scalar or vector
   # and returns the appropriate result

   def __mul__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return sum(other * w for w in self.vect)
       elif tp is Vector:
           return sum(v * w for v, w in zip(self.vect, other.vect))
       elif tp is list:
           return sum(v * w for v, w in zip(self.vect, other))
       else:
           pass

   def __imul__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return sum(other * w for w in self.vect)
       elif tp is Vector:
           return sum(v * w for v, w in zip(self.vect, other.vect))
       elif tp is list:
           return sum(v * w for v, w in zip(self.vect, other))
       else:
           pass

   def __truediv__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return Vector([xi / other for xi in self.vect])
       elif tp is Vector:
           pass

   def __idiv__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return Vector([xi / other for xi in self.vect])
       elif tp is Vector:
           pass

   # String representation of the object
   # for the print command
   def __repr__(self):
       return "Vector: " + str(self.vect)

   def add(one, other):
       return [xi + xt for xi, xt in zip(one, other)]

   def vector_sum(self, vectors):
       assert isinstance(vectors, object)
       return reduce(Vector.add, vectors)

   def vector_sub(vectors):
       assert isinstance(vectors, object)
       return reduce(self.__sub__, vectors)

   def scalar_prod(self, factor):
       return [factor * v for v in self.vect]

   def dot(self, other):
       return sum(v * w for v, w in zip(self.vect, other.vect)) if type(other) is Vector else sum(v * w for v, w in zip(self.vect, other))

   def sum_of_squares(self):
       return self.dot(self.vect, self.vect)

   def magnitude(self):
       return math.sqrt(self.sum_of_squares())


class Matrix:

   def __init__(self, matr):
       self.matr = Matrix.validate(matr)
       self.dim = self.dim()

   def validate(matr):
       if type(matr) is list:
           _, n = size(matr)
           for row in matr:
               if len(row) is n:
                   continue
               else:

           return matr

   def dim(self):
       return size(self.matr)

   def __add__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return Matrix([x + other for x in row] for row in self.matr)
       elif tp is Matrix:
           if self.dim() is other.dim():
               return Matrix([[xi + xt for xi, xt in zip(row1, row2)] for row1, row2 in zip(self.matr, other.matr)])
           else:
               raise Exception("Dimensions don't match")

   def __sub__(self):
       tp = type(other)
       if tp is int or tp is float:
           return Matrix([x - other for x in row] for row in self.matr)
       elif tp is Matrix:
           if self.dim() is other.dim():
               return Matrix([[xi - xt for xi, xt in zip(row1, row2)] for row1, row2 in zip(self.matr, other.matr)])
           else:
               raise Exception("Dimensions don't match")

   def __mul__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return Matrix([[xi * other for xi in row] for row in self.matr])
       elif tp is Matrix:
           return Matrix(matrix_mult(self.matr, other.matr))

   def __truediv__(self, other):
       tp = type(other)
       if tp is int or tp is float:
           return Matrix([x / other for x in row] for row in self.matr)
       else:
           return Matrix(self.matr * inv(other.matr))

   def hadamard(self, other):
       if type(other) is Matrix:
           return Matrix(Matrix([[xi * xt for xi, xt in zip(row1, row2)] for row1, row2 in zip(self.matr, other.matr)]))


   def determinant(self):
       pass

   def elim(self):
       pass

   def inv(self):
       pass

   def __invert__(self):
       pass

   def rem(self, offset):
       pass

   def remcol(self):
       return Matrix([row[1:] for row in self.matr])


"""
ZEROS function

Initializes a matrix of m x n of all zeros

"""

def zeros(m, n):
   return [[0 for _ in range(n)] for _ in range(m)]

   # In case you use British-English

def zeroes(m, n):
   return zeros(m, n)

"""
ONES function

Usage: ones(m, n)

Initializes a matrix of m x n of all ones

"""

def ones(m, n):
   return [[1 for _ in range(n)] for _ in range(m)]

"""
IDENTITY MATRIX FUNCTION

Usage: identity(n)

Returns an identity of matrix of size n x n
n must be greater than 1
"""

def identity(n):
   return [[1 if i == j else 0 for i in range(n)] for j in range(n)] if n > 1 else 1

"""

SIZE function

Returns the row/column size of a matrix/vector
in the format [# of rows, # of columns]

"""

def size(v):
   if type(v[0]) is int:
       c = 1
   elif type(v[0]) is list:
       c = len(v[0])

   return len(v), c

"""

MATRIX MULTIPLICATION FUNCTION

Multiplies two matrices a and b (list of lists) with size m x n and n x p
Returns a matrix of size m x p

Operator: a * b

Returns false if the multiplication is impossible

"""

def matrix_mult(a, b):
   if len(a[0]) == len(b):
       return [[sum(a[row][i] * b[i][col] for i in range(len(b))) for col in
                   range(len(b[0]))] for row in range(len(a))]
   else:
       return False

"""
TRANSPOSE function

Returns a transposed a vector / matrix v

"""

def transpose(v):
   rows, cols = size(v)

   return [[v[row] if type(v[row]) is int else v[row][col]
               for row in range(rows)] for col in range(cols)]

"""

rand function

returns a randomly generated m x n matrix
with values between 0 and 1

"""

def rand(m, n):
   return [[random.random() for _ in range(m)] for _ in range(n)]

"""

DISTANCE function

Distance function accepting two lists with [x, y, z...]
coordinates and returns the distance between them
using Pythagorean theorem

"""

def distance(one, other):
   return math.sqrt(sum((a - b) ** 2 for a, b in zip(one, other)))


def sigmoid(x):

   def logit(x):
       return 1 / (1 + math.exp(-x))

   tp = type(x)

   if tp is int or tp is float:
       return logit(x)
   elif tp is Vector:
       return Vector([logit(xi) for xi in x.vect])
   elif tp is Matrix:
       return Matrix([[logit(xi) for xi in row] for row in x.matr])


def sigmoid_grad(x):

   def logit(x):
       return 1 / (1 + math.exp(-x))

   def logit_grad(x):
       return logit(x) * (1 - logit(x))

   tp = type(x)

   if tp is int or tp is float:
       return logit_grad(x)
   elif tp is Vector:
       return Vector([logit_grad(xi) for xi in x.vect])
   elif tp is Matrix:
       return Matrix([[logit_grad(xi) for xi in row] for row in x.matr])


"""

NEURAL NETWORK CLASS

Implements a neural network with all the necessary functions and variables
including input layer, hidden layer, and output layer size and number,
initial theta, random initialization of theta, forward and backward propagation
algorithms, and a neural network training algorithm.

"""

class NeuralNetwork:

   """
   Usage:

   Network(struct, network)

   struct is a list detailing the layers of the network
   for example, with a three layer network with 20 input neurons,
   5 neurons in the hidden layer and one output neuron,
   struct = [20, 5, 1]

   network is a list of lists of lists representing the network's weights
   without the bias term


   """

   def __init__(self, struct=None, network=None):
       if network is None and type(struct) is list:
           self.network = self.initNetwork(struct)
       elif type(network) is list and struct is None:
           self.network = network
           self.struct = self.findStruct(network)
       elif network is None and struct is None:
           raise Exception("Networks cannot have both an undefined structure and weights!")
       else:
           self.network = network
           self.struct = struct


   # Randomly initializes weights for the network if its network
   # of weights is undefined

   def initNetwork(self, structure):

       def epsilon(l_in, l_out):
           return math.sqrt(6) / math.sqrt(l_in + l_out)

       network = []

       for input in structure[1:]:
           for output in structure[:-1]:
               eps = epsilon(input, output)
               network.append((2 * eps * Matrix(rand(output, input+1)) - eps).matr)

       return network


   def findStruct(self, network):

       struct = []

       for layer in network:
           if layer is network[-1]:
               m, n = dim(layer)
               struct.append(n-1)
               struct.append(m)

           _, n = dim(layer)
           struct.append(n-1)

       return struct


   def feed_forward(self, inputs, activ_func=None):

       if activ_func is None:
           activ_func = sigmoid

       outputs = []
       zs = []

       for layer in self.network:

           z = Matrix(inputs) * Matrix(transpose(layer))

           cur_output = activ_func(z)

           zs.append(z.matr)
           outputs.append(cur_output.matr)
           inputs = cur_output

       return outputs, zs

   def backpropagate(self, inputs, outputs, activ_function=None, grad_function=None):

       if activ_function is None and grad_function is None:
           activ_function = sigmoid
           grad_function = sigmoid_grad

       deltas = []
       gradients = []

       hypotheses, z_vals = self.feed_forward(inputs, activ_function)

       gradients.append((Matrix(hypotheses[-1]) - Matrix(outputs)).matr)

       network = self.network.sort(reverse=True)

       zs = z_vals.sort(reverse=True)

       counter = 0

       for layer in network:

           layer = Matrix(layer).remcol()

           delta_l = (Matrix(transpose(layer.matr)) * Matrix(gradients[counter])).hadamard(Matrix(grad_function(zs[counter+1]))).matr
           deltas.append(delta_l)

           counter = counter + 1

       return gradients


   def train(self, inputs, outputs, lr, iterations, activ_function=None, grad_function=None):

       for _ in range(iterations):

           gradients = self.backpropagate(inputs, outputs, activ_function, grad_function)

       return network


"""

K-MEANS CLUSTERING algorithm

An unsupervised learning algorithm that uses the average distance
between a centroid (center of a cluster) and a data point to
gradually form sensible clusters/groupings around unlabelled data.

"""

class KMeans:

   def __init__(self, k):
       self.k = k

   # Finds the centroids
   def train(self, data):

       # Assigns the point to its closest centroid
       def assign(point, centroids):
           distances = [distance(point, centroid) for centroid in centroids]
           return distances.index(min(distances))

       # Calculates the new distance of a centroid using its assigned points
       def calc_dist(assn_points):
           summative = Vector([0 for _ in range(len(assn_points[0]))])
           return (Vector(summative.vector_sum(assn_points)) / len(assn_points)).vect

       #Initializes random centroids from dataset
       centroids = random.choices(data, k=self.k)

       # Initializes distance arrays for comparison and usage
       prev_distances = [0 for _ in range(self.k)]
       distances = prev_distances

       # Adjusts centroids
       while True:

           # Assigns each point to a centroid based on its distance
           assigned_points = [assign(point, centroids) for point in data]

           # Calculates new distances for centroids
           distances = [calc_dist([point for point, assignment in zip(data, assigned_points) if assignment == centroid]) for centroid in range(self.k)]

           # Adjusts the centroids to the new computed distances
           centroids = distances

           # Checks if the distances are converging/not changing
           # if they are converging, stop the loop
           # else, continue and assign
           # the computed distances to be the next
           # previous distances

           if prev_distances == distances: break
           else: prev_distances = distances

       # Returns the locations of the clusters
       return distances

'''

K_NEAREST_NEIGHBORS algorithm

Premise: predict the label of the data point associated
with a certain coordinate assuming distance to another neighbor
is a factor in predicting labels

returns the predicted label or winner
for the inputted set of coordinates

'''
''' Expects self and other coordinates [x, y, z...] '''


def k_nearest_neighbors(k, data, prediction):

   def vote(nearest):

       vote_counts = Counter(nearest)
       winner, win_count = vote_counts.most_common(1)[0]

       wins = [count for count in vote_counts.values() if count == win_count]
       print(winner)

       if len(wins) > 1:
           vote(nearest[:-1])
       else:
           return winner

   data_with_distance = [(distance(prediction, coord), label) for label, coord in data]
   sorted_distance = sorted(data_with_distance, key=lambda x: x[0])

   k_nearest = sorted_distance[:k]

   return vote(k_nearest)



"""
BATCH GRADIENT DESCENT Algorithm

Expects x in list of lists arrangement to compensate for multiple regression
i.e [1, x11, x12, x13...] for x1
the 1 in the beginning stands for the constant term
since theta at 0 is the constant term

Expects y data in [y1, y2, y3...] format

Stops iterating when theta converges
i.e when the errors start becoming closer and closer to each other
the difference between previous and current errors
is below some tolerance value

Error function must accept theta, x_i, and y_i
Error function must return ONE instance of the error
for one (x, y) point versus the theta

This is also true of the gradient function; the summation
for the "batch" part of gradient descent is done
within the gradient descent function itself

Arguments: x, y, theta, the error function,
the gradient of the error function, the learning rate
(defaults to 0.0001 if none is specified), and the tolerance
to stop the iterations and converge (defaults to 0.00001), if
no iteration count is specified

If an iteration count is specified, the algorithm will iterate
the specified amount of times without regards to
convergence of the algorithm.

Returns the theta and a list of errors as the
function iterates for debugging.
Both are lists.

"""

def batch_GradientDescent(x, y, theta, error_func, gradient_func, iterations, lr=0.1, tolerance=0.001):
   # Built in step function to modify the theta based on the gradient and learning rate
   def step(theta, gradients, lr):
       return [theta_i - lr * gradient_i for theta_i, gradient_i in zip(theta, gradients)]

   #List of errors to be returned and plotted in the linear regression function
   #for cool visualization's sake

   errors = []

   # Checks if iterations were specified or not
   # and chooses the appropriate method
   # to ensure the algorithm stops
   # or converges

   for _ in range(iterations):
       # Creates a new Vector representation for the gradient
       gradients = Vector([0 for _ in range(len(x[0]))])

       # Sums all of the gradients to the 'gradients' vector
       gradients.vector_sum([gradient_func(theta, x_i, y_i, len(x)) for x_i, y_i in zip(x, y)])

       # Creates an updated theta based on the gradient and learning rate
       new_theta = step(theta, gradients.vect, lr)

       # Calculates the error after the theta is updated
       current_error = sum(error_func(new_theta, x_i, y_i, len(x)) for x_i, y_i, in zip(x, y))

       # Updates theta to the "new theta"
       theta = new_theta

       # Appends the error to the error list
       errors.append(current_error)

   return theta, errors

"""
STOCHASTIC GRADIENT DESCENT Algorithm

Expects x in list of lists arrangement to compensate for multiple regression
i.e [1, x11, x12, x13...] for x1
the 1 in the beginning stands for the constant term
since theta at 0 is the constant term

Expects y data in [y1, y2, y3...] format

Stops iterating when theta converges
i.e when the errors start becoming closer and closer to each other
the difference between previous and current errors
is below some tolerance value, unless an iteration
count is specified

If an iteration count is specified, the algorithm will iterate
the specified amount of times without regards to
convergence of the algorithm.

Error function must accept theta, x_i, and y_i
Error function must return ONE instance of the error
for one (x, y) point versus the theta

This is also true of the gradient function; the summation
for the "batch" part of gradient descent is done
within the gradient descent function itself

Arguments: x, y, theta, the error function,
the gradient of the error function, the learning rate
(defaults to 0.0001 if none is specified), and the tolerance
to stop the iterations and converge (defaults to 0.00001)

Returns the theta and a list of errors as the
function iterates for debugging.
Both are lists.

"""

def stochastic_GradientDescent(x, y, theta, error_func, gradient_func, iterations=0, lr=0.0001, tolerance=0.0001):
   def step(theta, gradients, lr):
       return [theta_i - lr * gradient_i for theta_i, gradient_i in zip(theta, gradients)]

   # List of errors to be returned and plotted in the linear regression function
   # for cool visualization's sake

   errors = []

   # Checks if iterations were specified or not
   # and chooses the appropriate method
   # to ensure the algorithm stops somehow

   if iterations:
       #Iterates over the specified number of iterations
       for _ in range(iterations):
           # Chooses a random point to be used "stochastically" for the
           # error and error gradient functions
           # to reduce computation number and time taken

           x_i = random.choice(x)
           y_i = random.choice(y)

           # Creates a new Vector representation for the gradient
           gradients = Vector(gradient_func(theta, x_i, y_i))

           # Creates an updated theta based on the gradient and learning rate
           new_theta = step(theta, gradients.vect, lr)

           # Calculates the error after the theta is updated
           current_error = sum(error_func(new_theta, x_i, y_i) for x_i, y_i, in zip(x, y))

           # Updates theta to the "actual theta"
           theta = new_theta

           # Appends the error to the error list
           errors.append(current_error)

   else:
       # Inputs a random point to be chosen as the initial error
       # To check for convergence in this case

       x_i = random.choice(x)
       y_i = random.choice(y)

       previous_error = error_func(theta, x_i, y_i)
       errors.append(previous_error)

       #Iterates until convergence
       while True:

           # Chooses a random point to be used "stochastically" for the
           # error and error gradient functions
           # to reduce computation number and time taken

           x_i = random.choice(x)
           y_i = random.choice(y)

           # Creates a new Vector representation for the gradient
           gradients = Vector(gradient_func(theta, x_i, y_i))

           # Creates an updated theta based on the gradient and learning rate
           new_theta = step(theta, gradients.vect, lr)

           # Calculates the error after the theta is updated
           current_error = sum(error_func(new_theta, x_i, y_i) for x_i, y_i, in zip(x, y))

           # Updates theta to the "actual theta"
           theta = new_theta

           # Appends the error to the error list
           errors.append(current_error)

           # Checks for convergence based on the tolerance
           if (abs(previous_error - current_error) < tolerance):
               break
           else:
               previous_error = current_error


   return theta, errors

#def miniBatch_GradientDescent


def linregTest(x, y, iterations=0, a=1, b=0, lr= 0.001):
   theta = [b, a]

   def error(theta, x, y):
       return y - Vector(theta).dot(Vector(theta))

   def mean_squared_error(theta, x, y, n):
       return error(theta, x, y) ** 2 / n

   def gradient(theta, x, y, n):
       return [-x[1] * error(theta, x, y) / n, -error(theta, x, y) / n]

   return batch_GradientDescent(x, y, theta, mean_squared_error, gradient, iterations, lr)

def linregStochastic(x, y, a=1, b=0):
   theta = [b, a]

   def error(theta, x, y):
       model = Vector(theta)
       x_i = Vector(x)
       return y - model.dot(x_i)

   def mean_squared_error(theta, x, y):
       return error(theta, x, y) ** 2

   def gradient(theta, x, y):
       return [-x[1] * error(theta, x, y), -error(theta, x, y)]

   return stochastic_GradientDescent(x, y, theta, mean_squared_error, gradient)

def linregStochastic_iter(x, y, iterations, a=1, b=0):
   theta = [b, a]

   def error(theta, x, y):
       model = Vector(theta)
       x_i = Vector(x)
       return y - model.dot(x_i)

   def mean_squared_error(theta, x, y):
       return error(theta, x, y) ** 2

   def gradient(theta, x, y):
       return [-x[1] * error(theta, x, y), -error(theta, x, y)]

   return stochastic_GradientDescent_iter(x, y, theta, mean_squared_error, gradient, iterations)

"""

Logistic Regression Algorithm

Maximizes the log probability of a sigmoid/logistic function
based on the assumption that the sigmoid function represents
the probability of success

Note that this type of logistic regression only has
two classes, and cannot classify more than two classes
For multinomial logistic regression, please refer
to the softmax regression implementation in neural networks

Uses batch gradient descent

"""

# Returns value of sigmoid or logistic function
# for a single x value
# definition of logistic function:
# e^x / e^x + 1 or math.exp(x) / (math.exp(x) + 1)
# dividing the numerator and denominator by e^x (equivalent to 1)
# will simplify to 1 / 1 + e^-x
# or 1 / (1 + math.exp(-x))
# much more efficient computation-wise

def logit(x):
   return 1 / (1 + math.exp(-x))

# Returns a prediction for binary logistic regression (i.e. 0 or 1)
# for a single x value

def predict(x, theta):
   return logit(Vector(theta).dot(x)) >= 0.5

def logitreg(x, y, theta, iterations=0):

   # Returns the cost function using the logistic function
   # for maximum likelihood estimation
   # based on the fact that the sigmoid function
   # is assumed to stand in as the probability
   # of belonging to the success class
   # assuming decision boundary of 0.5
   # since there are two classes, the probability distribution
   # is Binomial. The log_odds function returns the negative log of that
   # binomial distribution, since individual probabilities
   # that are independent, i.e. the probabilities of the
   # entire training set can be summed instead of
   # multiplied, easier computation and function-wise
   # it is negated so that instead of maximizing the likelihood
   # the negative of the likelihood can be minimized
   # using gradient descent or some normal equation (analytical)

   def log_odds(theta, x, y):
       return -y * math.log(logit(Vector(theta).dot(x))) - (1-y) * math.log(1-logit(Vector(theta).dot(x)))


   # The gradient of the log_odds function
   # Calculated using chain and product rule
   # Try it yourself! It simplifies nicely!

   def log_odds_gradient(theta, x, y):
       return [(logit(Vector(theta_i).dot(x_i)) - y) * x_i for x_i, theta_i in zip(x, theta)]

   return batch_GradientDescent(x, y, theta, log_odds, log_odds_gradient, iterations)

"""

ONE VS ALL LOGISTIC REGRESSION

Premise:

For multiple labels, logistic regression can still be used to classify labels
even with its binary (0 or 1) classification premise
by classifying one class as positive and everything else (the other labels)
as negative or non-matching.
Logistic regression is then used to train thetas for each of these classes
using the method outlined above

Accepts x like every other algorithm
Accepts y as a column vector of labels of positive integer numbers
corresponding to each row of x

(for example, [1, 2, 0, 1, 4, 5...])

Number of classes must be specified.

Returns a matrix of thetas; each row corresponds to the theta
of the class of that index (if classes start with 0)

"""

def OneVsAll(x, y, num_classes, iterations=0):

   # Function to preprocess y for all classes
   # Turning every other class to 0
   # and the selected class to 1

   def PreProcess(cls, y):
       return [1 if x == cls else 0 for x in y]

   # Initialize list of thetas
   thetas = []

   num_rows = len(x)
   num_cols = len(x[0])

   # Initialize random initial theta
   rand_theta = zeros(num_cols, 1)

   # Iterates over the classes and trains thetas for each of them
   for cls in range(num_classes):
      thetas[cls] = logitreg(x, PreProcess(cls, y), rand_theta, iterations)

   return thetas

"""

ONE VS ALL prediction funcion

Predicts the associated class by determining the maximum likelihood
of an entire batch of training examples belonging to a certain class

"""

def predictOneVsAll(X, thetas):

   predictions = matrix_mult(X, thetas)

   return [i for i, x in enumerate(predictions[row])
           if x == max(predictions[row]) for row in range(len(predictions))]

"""

Linear Regression Algorithm

Premise:

Predict the slope and y-intercept of a function given a data set
in two dimensions in that the error between the function
and the data set is the lowest possible value or minimum

a is the slope, b is the y-intercept or constant

Does not use implemented gradient descent function, rather
implements its own variation. That will change soon.

"""


def linreg(x, y, iterations, a=1, b=0, lr=0.0001):
   def error(a, b, x, y):
       return y - (a * x + b)

   def mean_squared_error(a, b, x, y, n):
       return error(a, b, x, y) ** 2 / (2 *n)

   def gradient(a, b, x, y, n):
       return [-error(a, b, x, y) / n, -x * error(a, b, x, y) / n]

   def step(theta, gradients, lr):
       return [theta_i - lr * gradient_i for theta_i, gradient_i in zip(theta, gradients)]

   theta = [b, a]
   err = []

   for _ in range(iterations):
       total_error = sum(mean_squared_error(a, b, x_i, y_i, len(x)) for x_i, y_i, in zip(x, y))

       gradients = Vector([0, 0])

       for x_i, y_i in zip(x, y):
           gradients.add(gradient(a, b, x_i, y_i, len(x)))

       new_theta = step(theta, gradients.vect, lr)

       theta = new_theta
       a, b = theta[1], theta[0]

       err.append(total_error)

   return theta, err



if __name__ == "__main__":
  # some messy testing code
  
   #k_data = [("Yes", [100, 100]), ("Yes", [90, 90]), ("Yes", [80, 80]), ("No", [0, 0]), ("No", [10, 10]),
             #("No", [20, 20]), ("No", [30, 30])]
   #print(k_nearest_neighbors(3, k_data, [50, 50]))

   train = KMeans(2)
   dataset = [[2, 1], [1, 1], [3, 2], [1.5, 1.5], [3, 3], [-2, -2], [-3, -2], [-3, -1.5], [-2, -2.5], [-1, -1]]

   #centroids = train.train(dataset)

   #assigned_points = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

   #sample = [point for point, assignment in zip(dataset, assigned_points)]



   #print(sample)

   #print(centroids)

   # if len(x) == len(y):
   #     theta, errors = linregTest(x, y)
   #     theta1, errors1 = linregStochastic(x, y)
   #     theta2, errors2 = linreg(x_2, y, 1000)
   #     theta3, errors3 = linregStochastic_iter(x, y)
   #
   #     thetas = [theta, theta1, theta2, theta3]
   #     errors = [errors, errors1, errors2, errors3]
   #
   # for theta_i, error_i in zip(thetas, errors):
   #     theta_vector = Vector(theta_i)
   #     print(theta_i)
   #
   #     model = [Vector(x_i).dot(theta_vector) for x_i in x]
   #
   #     plt.plot(y, 'r')
   #     plt.plot(model, 'b')
   #     plt.title("Model vs Actual")
   #     plt.show()
   #
   #     plt.plot(error_i, 'r')
   #     plt.title("errors")
   #     plt.show()

  

