import pickle
import numpy as np
import os
import time
import tensorflow as tf

from .cleverhans_wrapper import generate_fgsm_adversarial_example, generate_pgd_adversarial_example


def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, x_adv_cache_path,
                                attack_log_fpath=None):
    if os.path.isfile(x_adv_cache_path):
        print("Loading adversarial examples from [%s]." % os.path.basename(x_adv_cache_path))
        X_adv, duration = pickle.load(open(x_adv_cache_path, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_adv_examples(model, x, y, X, Y, attack_name, attack_params, attack_log_fpath)
        duration = time.time() - time_start

        # 如果 X_adv 是 Tensor，转换为 numpy array
        if isinstance(X_adv, tf.Tensor):
            # 对于 TensorFlow 2.x
            if hasattr(X_adv, 'numpy'):
                X_adv = X_adv.numpy()
            else:
                # TensorFlow 1.x 需要使用 session 来转换
                X_adv = sess.run(X_adv)

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

    return X_adv, duration


def generate_adv_examples(model, x, y, X, Y, attack_name, attack_params, attack_log_fpath):
    if attack_name == 'none':
        return X
    elif attack_name == 'fgsm':
        return generate_fgsm_adversarial_example(model_fn=model, input_tensor=X)
    elif attack_name == 'pgd':
        return generate_pgd_adversarial_example(model_fn=model, input_tensor=X)
    else:
        raise NotImplementedError("Unsuported attack [%s]." % attack_name)
