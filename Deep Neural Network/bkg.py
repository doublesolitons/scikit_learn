'''
Perceptron algorithm
     multiple inputs and one output. z = sum(W * X) + b
     activation function: sign(z)

Deep Neural Network (DNN) model
    inputs:
        number of layers
        number of neurons at each hidden and output layer
        activation function
        loss function
        learning rate
        max iterations
        termination threshold
        training samples
    output:
        weight matrix and bias at each hidden and output layer

Back propagation pseudocode
    1: initiate weight matrix and bias
    2: for iter from 1 to MAX:
        i: for i = 1 to m:
            a) a(1) = X_{i}
            b) for l = 2 to L:
                a(l)_{i} = f(z(l)_{i}) = f(W(l) * a(l - 1)_{i} + b(l))
            c) calculate sigma(L)_{i} based on loss function J_{i}
            d) for l = L - 1 to 2:
                calculate sigma(l)_{i}
        ii: for l = 2 to L:
            a) update W(l), b(l)
        iii: if delta(W) < epsilon & delta(b) < epsilon for all W and b:
            a) break
    3: output W and b


'''