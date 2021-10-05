"""
Can be used on supervised or unsupervised data
Supervised KNN:
    Classification: Discrete labels
    Regression:     Continuous labels

Metrics: Euclidean distance
Algorithms:
    Brute Force:    M samples, N dimensions. Complexity O[N * M^^2]
    K-D Tree:       works well for low-dimension data
    K-D Ball:       Works well for high-dimension data
    Selection of algorithms:
        Query time: Brute force --> O[N * M]
                    Ball tree   --> O[N * log(M)]
                    KD tree     --> O[N * log(M)] for less than 20 features; O[N * M] for large features
                    if M < 30, brute force is preferred than tree-based approach

"""