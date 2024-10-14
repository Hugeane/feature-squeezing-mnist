# Attack Algorithms

## Default Parameters

### Cleverhans Attacks
 
* eps: (required float) maximum distortion of adversarial example compared to original input
* eps_iter: (required float) step size for each attack iteration
* nb_iter: (required int) Number of attack iterations.
* theta: (optional float) Perturbation introduced to modified components (can be positive or negative)
* gamma: (optional float) Maximum percentage of perturbed features

|  Parameter |  FGSM   | PGD  |
|------------|---------|------|
|    eps     |  0.1    | 0.1  |
|  eps_iter  |   -   | 0.05 |
|  nb_iter   |   -   | 10   |
|   theta    |   -   | -    |
|   gamma    |   -   | -    |

### DeepFool

* num_classes: limits the number of classes to test against.
* overshoot: used as a termination criterion to prevent vanishing updates 
* max_iter: maximum number of iterations for deepfool

Example: `python --attacks "deepfool?overshoot=9&max_iter=50"`