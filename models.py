import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        
        return nn.DotProduct(x, self.w)    
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                y_val = nn.as_scalar(y) 
                if self.get_prediction(x) != y_val:
                    self.w.update(x, y_val)
                    converged = False
        
        
        
        
        
        
        
        
        
        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        
        self.hidden_size = 512        
        self.lr = 0.05    

        self.W1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)

        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        h  = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        y_hat = nn.AddBias(nn.Linear(h, self.W2), self.b2)
        return y_hat  
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        
        return nn.SquareLoss(self.run(x), y)
        
        

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size   = 20 
        target_loss  = 0.02
        check_every  = 50
        steps        = 0

        for x, y in dataset.iterate_forever(batch_size):
            loss   = self.get_loss(x, y)
            grads  = nn.gradients(loss,
                     [self.W1, self.b1, self.W2, self.b2])
            for p, g in zip([self.W1, self.b1, self.W2, self.b2], grads):
                p.update(g, -self.lr)

            steps += 1
            if steps % check_every == 0:
                full_loss = nn.as_scalar(self.get_loss(
                             nn.Constant(dataset.x),
                             nn.Constant(dataset.y)))
                if full_loss < target_loss:
                    break
        

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        self.hidden_size = 200
        self.lr          = 0.5

        self.W1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1,   self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1,   10)
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        h      = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        logits = nn.AddBias(nn.Linear(h, self.W2), self.b2)
        return logits
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        
        return nn.SoftmaxLoss(self.run(x), y)
        

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size   = 100
        target_acc   = 0.975
        max_epochs   = 20

        for _ in range(max_epochs):
            for x, y in dataset.iterate_once(batch_size):
                loss   = self.get_loss(x, y)
                grads  = nn.gradients(loss,
                         [self.W1, self.b1, self.W2, self.b2])
                for p, g in zip([self.W1, self.b1, self.W2, self.b2], grads):
                    p.update(g, -self.lr)

            if dataset.get_validation_accuracy() >= target_acc:
                break
        
        
        
        
        
        
        
        
        
        

class LanguageIDModel(object):
    
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        
        self.hidden_size = 384
        self.lr          = 0.12

        self.Wx = nn.Parameter(self.num_chars, self.hidden_size)
        self.Wh = nn.Parameter(self.hidden_size, self.hidden_size)
        self.bh = nn.Parameter(1, self.hidden_size)

        self.Wo = nn.Parameter(self.hidden_size, len(self.languages))
        self.bo = nn.Parameter(1, len(self.languages))
        
    def _step(self, x_t, h_prev):
        z = nn.Add(nn.Linear(x_t,  self.Wx),
                   nn.Linear(h_prev, self.Wh))
        z = nn.AddBias(z, self.bh)
        return nn.ReLU(z)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        
        batch = xs[0].data.shape[0]
        h = nn.Constant(np.zeros((batch, self.hidden_size)))

        for x_t in xs:
            z  = nn.Add(nn.Linear(x_t, self.Wx), nn.Linear(h, self.Wh))
            h  = nn.ReLU(nn.AddBias(z, self.bh))

        logits = nn.AddBias(nn.Linear(nn.ReLU(h), self.Wo), self.bo)
        return logits
              

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        
        return nn.SoftmaxLoss(self.run(xs), y)
        
            
    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size  = 64
        max_epochs  = 60
        target_acc  = 0.82

        best_acc   = 0.0
        best_state = None

        for _ in range(max_epochs):
            for xs, y in dataset.iterate_once(batch_size):
                loss   = self.get_loss(xs, y)
                grads  = nn.gradients(loss,
                         [self.Wx, self.Wh, self.bh, self.Wo, self.bo])
                for p, g in zip([self.Wx, self.Wh, self.bh, self.Wo, self.bo],
                                grads):
                    p.update(g, -self.lr)

            val = dataset.get_validation_accuracy()
            if val > best_acc:
                best_acc = val
                best_state = [p.data.copy() for p in
                              (self.Wx, self.Wh, self.bh, self.Wo, self.bo)]
                if best_acc >= target_acc:     
                    break

        for p, saved in zip([self.Wx, self.Wh, self.bh, self.Wo, self.bo],
                            best_state):
            p.data[:] = saved