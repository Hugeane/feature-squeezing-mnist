from models.carlini_models import carlini_mnist_model
from models.cleverhans_models import cleverhans_mnist_model
from models.pgdtrained_models import pgdtrained_mnist_model

import os
import tarfile
import urllib3

def load_model_by_name(self, model_name, logits=False, input_range_type=1, pre_filter=lambda x: x):
    """
    :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
    :params input_range_type: {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
    """
    if model_name not in ["cleverhans", 'cleverhans_adv_trained', 'carlini', 'pgdtrained', 'pgdbase']:
        raise NotImplementedError("Undefined model [%s] for %s." % (model_name, self.dataset_name))
    self.model_name = model_name

    download_dir = 'downloads'
    trained_model_dir = 'downloads/trained_models'

    # dir not exist, make dir and download trained models from github, then unzip the url file to directory downloads
    # github url: https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/downloads.tar.gz
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)
        # 下载和解压模型
        model_archive_url = "https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/downloads.tar.gz"
        model_archive_path = os.path.join(download_dir, 'downloads.tar.gz')
        if not os.path.isfile(model_archive_path):
            print("Downloading trained models...")
            http = urllib3.PoolManager()
            response = http.request('GET', model_archive_url)

            with open(model_archive_path, 'wb') as f:
                f.write(response.data)
            print("Download completed.")

            # 解压 tar.gz 文件
            with tarfile.open(model_archive_path, 'r:gz') as tar:
                tar.extractall(path=download_dir)
            print("Extraction completed.")

    model_weights_fpath = "%s_%s.keras_weights.h5" % (self.dataset_name, model_name)
    model_weights_fpath = os.path.join(trained_model_dir, model_weights_fpath)

    # self.maybe_download_model()
    if model_name in ["cleverhans", 'cleverhans_adv_trained']:
        model = cleverhans_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
    elif model_name in ['carlini']:
        model = carlini_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
    elif model_name in ['pgdtrained', 'pgdbase']:
        model = pgdtrained_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
    print("\n===Defined TensorFlow model graph.")
    model.load_weights(model_weights_fpath)
    print("---Loaded MNIST-%s model.\n" % model_name)
    return model