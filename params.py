"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# params for source dataset
src_dataset = "MNIST"
src_encoder_restore = "snapshots/ADDA-source-encoder-100.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-100.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "USPS"
# tgt_encoder_restore = "snapshots/ADDA-target-encoder-150.pt"
tgt_encoder_restore = None
tgt_model_trained = False

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 1
# d_model_restore = "snapshots/ADDA-critic-150.pt"
d_model_restore = None

# params for training network
num_gpu = 1
num_epochs_pre = 100
log_step_pre = 20
eval_step_pre = 5
save_step_pre = 20
num_epochs = 500
log_step = 5
save_step = 50
manual_seed = None

# params for optimizing models
d_learning_rate = 5e-5
c_learning_rate = 8e-5
beta1 = 0.5
beta2 = 0.9
