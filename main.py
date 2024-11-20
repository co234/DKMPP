import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from absl import app, flags
from model.data import STPPDataModule
from model.dkmpp import DKMPP
import torch
import scipy
import pandas as pd
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from torchmetrics.functional import mean_squared_error
import warnings
import pickle
import scipy.stats
warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS



# training flags
#flags.DEFINE_string("dataset", "data/synthetic_data/simulated_stpp_f1_u_deep_kernel_no_z.csv", "path to the dataset")

# Initial set up
flags.DEFINE_string("dataset", "synthetic", "dataset name")
# flags.DEFINE_string("dataset", "vancouver", "dataset name")
flags.DEFINE_string("loss_type", "dsm", "type of loss func")
flags.DEFINE_string("device", "cpu", "device")
flags.DEFINE_string("checkpoint", None, "checkpoint to load")
flags.DEFINE_string("kernel_type", "dkf", "type of kernel used")
flags.DEFINE_string("base_kernel", "rbf", "type of base kernel of deep kernel")
flags.DEFINE_string('log_file', None, '')

# Hyperparameter
flags.DEFINE_float("lr", 0.0001, "learning rate")
flags.DEFINE_float("l2", 1e-5, "l2 coefficient")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_integer("epochs", 10, "no. of epochs")
flags.DEFINE_integer("d_t", 32, "d_t")
flags.DEFINE_integer("d", 4, "input data dimensionality")
flags.DEFINE_integer("num_samples", 500000, "number of samples")
flags.DEFINE_float("prop", 1, "proportion of data used")
flags.DEFINE_float('dsm_sigma', 6.5, '')

# flags.DEFINE_bool('rbf', True, '')


def train(model, dataset):
    logger = TensorBoardLogger(
        "lightning_logs", name=f"dkmpp_{FLAGS.d}d_{FLAGS.loss_type}"
    )
    timer = Timer()
    # training
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        inference_mode=False,
        max_epochs=FLAGS.epochs,
        logger=logger,
        accelerator=FLAGS.device,
        devices=1,
        gradient_clip_val=1,
        detect_anomaly=True,
        callbacks=[timer, EarlyStopping(monitor="val_loss_epoch", mode="min", patience=10)]
    )

    dataset.setup()
    #test_dataloader = dataset.test_dataloader()
    trainer.fit(model, datamodule=dataset)

    print(f'training time: {timer.time_elapsed("train")}')
    trainer.validate(model,datamodule=dataset)
    trainer.test(model,datamodule=dataset)

    return timer.time_elapsed("train"), trainer.current_epoch

def eval(model):

    if FLAGS.dataset == 'synthetic':
         #=========PRINT PARAMETER INFORMATION==============
        print("------------ESTIMATED PARAMS INFO---------------")
        print("=>ESTIMATED PARAMS FOR f1: ")
        print("| w1 | {:.3f} |".format(model.weights_raw.view(-1).detach().numpy()[0]))
        print("| b1 | {:.3f} |".format(model.bias_raw.view(-1).detach().numpy()[0]))
        print("=>ESTIMATED PARAMS FOR f2: ")
        print("w2: {}".format(model.weights_kernel_1.view(-1).detach().numpy()))
        print("b2: {}".format(model.bias_kernel_1.view(-1).detach().numpy()))
        #print(model.weights_kernel_1.view(-1))
        #print(model.bias_kernel_1)
        print("=>EVALUATION FOR f2: ")
        w = model.weights_kernel_1.view(-1)
        x = torch.diag(torch.from_numpy(np.array([0.99,0.98,0.97]))).view(-1)
        metric = mean_squared_error(x,w)
        print("|  mse  | {:.2f} |".format(metric))
        print("| gamma | {:.2f} |".format(model.gamma))


        params_dict = {'w1':model.weights_raw.view(-1),
                       'b1':model.bias_raw,
                       'w2': model.weights_kernel_1.view(-1),
                       'b2': model.bias_kernel_1,
                       'lam': model.lamda,
                       'gamma': model.gamma}

        param_file = f"data/estimated_params_{FLAGS.d}d_{FLAGS.loss_type}_{FLAGS.prop}_{FLAGS.num_samples}_{FLAGS.dsm_sigma}.pkl"

        with open(param_file, 'wb') as f:
            pickle.dump(params_dict, f)

        points = torch.cartesian_prod(
            torch.arange(0, 101, 2),  # x
            torch.arange(0, 101, 2),  # y,
            torch.arange(0, 101, 2),  # t
        )
        z = torch.tensor(
            (
                scipy.stats.norm(50, 100).pdf(points[:, 0])
                + scipy.stats.norm(50, 200).pdf(points[:, 1])
            )
            * 10000,
            dtype=torch.float32
        )
        rp = torch.cat([points, z.unsqueeze(1)], dim=1)
        model.eval()
        intensities = model.intensity(rp.unsqueeze(0))
        gt_intensities = model.gt_intensity(rp.unsqueeze(0))

        rmse = torch.sqrt(mean_squared_error(intensities,gt_intensities))
        print("=================================================")
        print("Method: {}, Epoch size: {}, #MC samples: {}".format(FLAGS.loss_type, FLAGS.batch_size, FLAGS.num_samples))
        print("RMSE: {:.3f} ".format(rmse.detach().numpy()))
        print("=================================================")

        df = pd.DataFrame(
            {
                "x": rp[:, 0].numpy(),
                "y": rp[:, 1].numpy(),
                "t": rp[:, 2].numpy(),
                "intensity": intensities[0].data.numpy(),
            }
        )

    elif FLAGS.dataset == 'vancouver':
        df = pd.read_csv("data/real_data/vancouver/test.csv")

        data = torch.tensor(np.array(df),dtype=torch.float32)
        model.eval()
        intensities = model.intensity(data.unsqueeze(0))

        df['intensity'] = intensities[0].data.numpy()
        test_log_l = model.log_likelihood(data.unsqueeze(0), batch_size=len(data.unsqueeze(0)))
        print(test_log_l)

    return df


def main(argv):
    # data
    # 'data/simulation_1d_100.pkl'
    if len(argv) != 2:
        print('need to specify train or evaluate;\nexmaple: python main.py train')
        exit(1)

    SAVE_FILE = f"data/estimated_intensity_{FLAGS.d}d_{FLAGS.loss_type}_{FLAGS.prop}_{FLAGS.num_samples}_{FLAGS.dsm_sigma}.csv"
    dataset = STPPDataModule(
        batch_size=FLAGS.batch_size, data_prop=FLAGS.prop, data_name=FLAGS.dataset
    )
    # dataset.setup()
    # test_data = dataset.test_dataloader()
    # print(test_data.shape)


    # model
    if FLAGS.checkpoint is not None:
        model = DKMPP.load_from_checkpoint(FLAGS.checkpoint)
    else:
        model = DKMPP(
            d=FLAGS.d,
            d_t=FLAGS.d_t,
            lr=FLAGS.lr,
            l2=FLAGS.l2,
            num_samples=FLAGS.num_samples,
            loss_type=FLAGS.loss_type,
            kernel_type = FLAGS.kernel_type,
            base_kernel = FLAGS.base_kernel,
            dsm_sigma=FLAGS.dsm_sigma,
            dataset = FLAGS.dataset
        )

    if argv[1] == 'train':
        train_time, epochs = train(model, dataset)
        # df = eval(model)
        # df.to_csv(SAVE_FILE, index=False)
        # # save training data
        # if FLAGS.log_file is not None:
        #     with open(FLAGS.log_file, 'a+') as f:
        #         f.write(f'{FLAGS.loss_type},{FLAGS.lr},{FLAGS.batch_size},{epochs},{FLAGS.num_samples},{FLAGS.prop},{FLAGS.kernel_type},{train_time/epochs},{SAVE_FILE},{FLAGS.dsm_sigma}\n')

    elif argv[1] == 'evaluate':
        df = eval(model)
        df.to_csv(SAVE_FILE, index=False)
    else:
        print('need to specify train or evaluate;\nexmaple: python main.py train')
        exit(1)

if __name__ == "__main__":
    app.run(main)
