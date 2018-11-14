import random

#   Single layer perceptron
class slp:
    def __init__(self, num_of_inputs):
        self.eta = 0.25
        self.bias = 1.0
        self.weights = [0.01, 0.04, 0.02]

    def weight_adjust(self, inputs, target, output):
        # tries to get the correct outcome.

        self.weights[0] += self.eta * (target - output)
        for i in range(len(inputs)):
            self.weights[i+1] += self.eta * (target - output) * inputs[i]
        return None

    def output(self, input):

        out = self.bias * self.weights[0]
        for i in range(len(input)):
            out += input[i] * self.weights[i+1]
        return out

    def train(self, inputs, targets):

        for _ in range(1000):
            rand = random.randrange(len(inputs))
            out = self.output(inputs[rand])
            self.weight_adjust(inputs[rand], targets[rand], out)
        return None

if __name__ == "__main__":
    # inputs
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    inputs_NOT = [[0], [1]]
    # outputs
    target_AND = [-1,-1,-1,1]   # 0 0 0 1
    target_OR = [-1, 1, 1, 1]   # 0 1 1 1
    target_NOT = [-1, 1]        # 0 1
    target_XOR = [-1, 1, 1, -1] # 0 1 1 0  <- should not work.

    rob = slp(len(inputs[0]))
    rob.train(inputs, target_AND)
    print("0 AND 0: " + str(rob.output([0, 0])))
    print("0 AND 1: " + str(rob.output([0, 1])))
    print("1 AND 0: " + str(rob.output([1, 0])))
    print("1 AND 1: " + str(rob.output([1, 1])))

    rob = slp(len(inputs[0]))
    rob.train(inputs, target_OR)
    print("\n0 OR 0: " + str(rob.output([0, 0])))
    print("0 OR 1: " + str(rob.output([0, 1])))
    print("1 OR 0: " + str(rob.output([1, 0])))
    print("1 OR 1: " + str(rob.output([1, 1])))

    rob = slp(len(inputs_NOT[0]))
    rob.train(inputs_NOT, target_NOT)
    print("\n NOT 0: " + str(rob.output([0])))
    print("NOT 1: " + str(rob.output([1])))

    rob = slp(len(inputs[0]))
    rob.train(inputs, target_XOR)
    print("\n0 XOR 0: " + str(rob.output([0, 0])))
    print("0 XOR 1: " + str(rob.output([0, 1])))
    print("1 XOR 0: " + str(rob.output([1, 0])))
    print("1 XOR 1: " + str(rob.output([1, 1])))
