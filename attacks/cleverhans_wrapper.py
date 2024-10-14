import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


def generate_fgsm_adversarial_example(model_fn, input_tensor):
    return fast_gradient_method(model_fn=model_fn, x=input_tensor, eps=0.1, norm=np.inf, clip_min=0, clip_max=1)


def generate_pgd_adversarial_example(model_fn, input_tensor):
    return projected_gradient_descent(model_fn=model_fn, x=input_tensor, eps=0.1, eps_iter=0.05, nb_iter=10,
                                      norm=np.inf,
                                      clip_min=0,
                                      clip_max=1)
