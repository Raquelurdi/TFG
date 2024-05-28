from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict
import argparse
import os

# from math import log
import multiprocessing
import logging

from utils.metrics import levenshtein


class Arguments(object):
    """
    """

    def __init__(self, logger=None):
        """
        """
        self.logger = logger or logging.getLogger(__name__)
        parser_description = """
        NN Implentation for Layout Analysis
        """
        self.parser = argparse.ArgumentParser(
            description=parser_description,
            fromfile_prefix_chars="@",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.parser.convert_arg_line_to_args = self._convert_file_to_args
        # ----------------------------------------------------------------------
        # ----- Define general parameters
        # ----------------------------------------------------------------------
        general = self.parser.add_argument_group("General Parameters")
        general.add_argument(
            "--config", default=None, type=str, help="Use this configuration file"
        )
        general.add_argument(
            "--exp_name",
            default="carabela",
            type=str,
            help="""Name of the experiment. Models and data 
                                               will be stored into a folder under this name""",
        )

        general.add_argument(
            "--model",
            default="MLP",
            type=str,
            help="""MLP | LSTM""",
        )

        general.add_argument(
            "--path_file_groups",
            default="",
            type=str,
            help="""""",
        )
        
        general.add_argument(
            "--steps",
            default='50,50',
            type=str,
            help="Layers for the FF or RNN network",
        )
        general.add_argument(
            "--gamma_step",
            default=0.5,
            type=float,
            help="Layers for the FF or RNN network",
        )

        general.add_argument(
            "--DO",
            default=0.5,
            type=float,
            help="Drop out",
        )

        general.add_argument(
            "--work_dir", default="./work/", type=str, help="Where to place output data"
        )
        general.add_argument(
            "--type", default="sum2", type=str, help="sum or max"
        )
        # --- Removed, input data should be handled by {tr,val,te,prod}_data variables
        # general.add_argument('--data_path', default='./data/',
        #                     type=self._check_in_dir,
        #                     help='path to input data')
        general.add_argument(
            "--log_level",
            default="INFO",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )
        # general.add_argument('--baseline_evaluator', default=baseline_evaluator_cmd,
        #                     type=str, help='Command to evaluate baselines')
        general.add_argument(
            "--num_workers",
            default=0,
            type=int,
            help="""Number of workers used to proces 
                                  input data. If not provided all available
                                  CPUs will be used.
                                  """,
        )

        general.add_argument(
            "--loss",
            default="cross-entropy",
            type=str,
            help="""[CR,cross-entropy,crossentropy] | """,
        )

        general.add_argument(
            "--openset",
            default="",
            type=str,
            help="""[1vsall] | """,
        )

        general.add_argument(
            "--gpu",
            default=0,
            type=int,
            help=(
                "GPU id. Use -1 to disable. " "Only 1 GPU setup is available for now ;("
            ),
        )
        general.add_argument(
            "--seed",
            default=5,
            type=int,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--te_size",
            default=0.2,
            type=float,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--kfold",
            default=15,
            type=int,
            help="Numbers of kfold",
        )

        general.add_argument(
            "--layers",
            default='32,64,128',
            type=str,
            help="Layers for the FF or RNN network",
        )

        general.add_argument(
            "--classes",
            default='P,CP,O,A,T',
            type=str,
            help="Classes to use",
        )


        # ----------------------------------------------------------------------
        # ----- Define processing data parameters
        # ----------------------------------------------------------------------
        data = self.parser.add_argument_group("Data Related Parameters")

        # ----------------------------------------------------------------------
        # ----- Define dataloader parameters
        # ----------------------------------------------------------------------
        loader = self.parser.add_argument_group("Data Loader Parameters")
        loader.add_argument(
            "--batch_size", default=32, type=int, help="Number of images per mini-batch"
        )

        # ----------------------------------------------------------------------
        # ----- Define NN parameters
        # ----------------------------------------------------------------------
        net = self.parser.add_argument_group("Neural Networks Parameters")


        # ----------------------------------------------------------------------
        # ----- Define Optimizer parameters
        # ----------------------------------------------------------------------
        optim = self.parser.add_argument_group("Optimizer Parameters")
        optim.add_argument(
            "--optim",
            default="ADAM",
            type=str,
            choices=["ADAM", "SGD", "RMSprop"],
            help="""Choose the optimizer""",
        )
        optim.add_argument(
            "--alpha",
            default=0.1,
            type=float,
            help="MNB alpha",
        )
        optim.add_argument(
            "--lr",
            default=0.1,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--adam_beta1",
            default=0.5,
            type=float,
            help="First ADAM exponential decay rate",
        )
        optim.add_argument(
            "--nheads",
            default=8,
            type=int,
            help="Transformer parameter",
        )
        optim.add_argument(
            "--nlayers",
            default=8,
            type=int,
            help="Transformer parameter",
        )
        optim.add_argument(
            "--type_seq",
            default="MLP",
            type=str,
            choices=["MLP", "mha", "tencoder", "encoder", "transformer"],
            help="""Choose the kind of """,
        )
        optim.add_argument(
            "--dim_feedforward",
            default=2048,
            type=int,
            help="Transformer parameter",
        )
        optim.add_argument(
            "--GSF",
            default=0.5,
            type=float,
            help="First ADAM exponential decay rate",
        )
        optim.add_argument(
            "--adam_beta2",
            default=0.999,
            type=float,
            help="Secod ADAM exponential decay rate",
        )
        # ----------------------------------------------------------------------
        # ----- Define Train parameters
        # ----------------------------------------------------------------------
        train = self.parser.add_argument_group("Training Parameters")
        tr_meg = train.add_mutually_exclusive_group(required=False)
        tr_meg.add_argument(
            "--do_train", dest="do_train", action="store_true", help="Run train stage"
        )
        tr_meg.add_argument(
            "--no-do_train",
            dest="do_train",
            action="store_false",
            help="Do not run train stage",
        )
        tr_meg.set_defaults(do_train=True)
        train.add_argument(
            "--num_train",
            default=100,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--num_test",
            default=1,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--save_rate",
            default=30,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--tr_data",
            default="/data/carabela/prev2/carabela_IG10KWords-SumScore.inf",
            type=str,
            help="""Train data folder. Train images are
                                   expected there, also PAGE XML files are
                                   expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--prod_data",
            default="/data/carabela/prev2/carabela_IG10KWords-SumScore.inf",
            type=str,
            help="""Train data folder. Train images are
                                   expected there, also PAGE XML files are
                                   expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--checkpoint_load",
            default="",
            type=str,
            help="""Train data folder. Train images are
                                   expected there, also PAGE XML files are
                                   expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--te_data",
            default="/data/carabela/index_viejos/vector_tfidf_max_test",
            type=str,
            help="""test data folder. """,
        )
        train.add_argument(
            "--class_dict",
            default="/data/carabela/index_viejos/vector_tfidf_max_test",
            type=str,
            help="""test data folder. """,
        )
        train.add_argument(
            "--LOO",
            default="no",
            type=self._str_to_bool,
            help="""Leaving one out option""",
        )
        train.add_argument(
            "--auto_lr_find",
            default="no",
            type=self._str_to_bool,
            help="""Leaving one out option""",
        )
        train.add_argument(
            "--do_prod",
            default="no",
            type=self._str_to_bool,
            help="""Leaving one out option""",
        )
        train.add_argument(
            "--do_test",
            default="no",
            type=self._str_to_bool,
            help="""Leaving one out option""",
        )
        train.add_argument(
            "--all_files",
            default="no",
            type=self._str_to_bool,
            help="""Leaving one out option""",
        )
        train.add_argument(
            "--IG_file",
            default="/data/carabela/index_viejos/vector_tfidf_max_test",
            type=str,
            help="""IG_file for SHAP """,
        )
        train.add_argument(
            "--tr_data_sum",
            default="/data/carabela/carabela_MACR-SumScore.inf",
            type=str,
            help="""Train data folder. Train images are
                                           expected there, also PAGE XML files are
                                           expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--epochs", default=10, type=int, help="Number of training epochs"
        )
        train.add_argument(
            "--num_feats", default=10000, type=int, help="Number of training epochs"
        )
        train.add_argument(
            "--weight_const",
            default=1.02,
            type=float,
            help="weight constant to fix class imbalance",
        )
        # ----------------------------------------------------------------------
        # ----- Define Test parameters
        # ----------------------------------------------------------------------
        test = self.parser.add_argument_group("Test Parameters")



    def _convert_file_to_args(self, arg_line):
        return arg_line.split(" ")

    def _str_to_bool(self, data):
        """
        Nice way to handle bool flags:
        from: https://stackoverflow.com/a/43357954
        """
        if data.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif data.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def _check_out_dir(self, pointer):
        """ Checks if the dir is wirtable"""
        if os.path.isdir(pointer):
            # --- check if is writeable
            if os.access(pointer, os.W_OK):
                if not (os.path.isdir(pointer + "/checkpoints")):
                    os.makedirs(pointer + "/checkpoints")
                    self.logger.debug(
                        "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                    )
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not writeable.".format(pointer)
                )
        else:
            try:
                os.makedirs(pointer)
                self.logger.debug("Creating output dir: {}".format(pointer))
                os.makedirs(pointer + "/checkpoints")
                self.logger.debug(
                    "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                )
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    "{} folder does not exist and cannot be created\n{}".format(e)
                )

    def _check_in_dir(self, pointer):
        """check if path exists and is readable"""
        if os.path.isdir(pointer):
            if os.access(pointer, os.R_OK):
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not readable.".format(pointer)
                )
        else:
            raise argparse.ArgumentTypeError(
                "{} folder does not exists".format(pointer)
            )

    def _check_to_int_array(self, data):
        """check is size is 256 multiple"""
        data = int(data)
        if data > 0 and data % 256 == 0:
            return data
        else:
            raise argparse.ArgumentTypeError(
                "Image size must be multiple of 256: {} is not".format(data)
            )


    def shortest_arg(self, arg):
        """
        search for the shortest valid argument using levenshtein edit distance
        """
        d = {key: (1000, key) for key in arg}
        for k in vars(self.opts):
            for t in arg:
                l = levenshtein(k, t)
                if l < d[t][0]:
                    d[t] = (l, k)
        return ["--" + d[k][1] for k in arg]

    def parse(self):
        """Perform arguments parsing"""
        # --- Parse initialization + command line arguments
        # --- Arguments priority stack:
        # ---    1) command line arguments
        # ---    2) config file arguments
        # ---    3) default arguments
        self.opts, unkwn = self.parser.parse_known_args()
        if unkwn:
            msg = "unrecognized command line arguments: {}\n".format(unkwn)
            msg += "do you mean: {}\n".format(self.shortest_arg(unkwn))
            self.parser.error(msg)

        # --- Parse config file if defined
        if self.opts.config != None:
            self.logger.info("Reading configuration from {}".format(self.opts.config))
            self.opts, unkwn_conf = self.parser.parse_known_args(
                ["@" + self.opts.config], namespace=self.opts
            )
            if unkwn_conf:
                msg = "unrecognized  arguments in config file: {}\n".format(unkwn_conf)
                msg += "do you mean: {}\n".format(self.shortest_arg(unkwn_conf))
                msg += "In the meanwile, solve this maze:\n"
                self.parser.error(msg)
            self.opts = self.parser.parse_args(namespace=self.opts)
        # --- Preprocess some input variables
        # --- enable/disable
        self.opts.use_gpu = self.opts.gpu != -1

        layers = self.opts.layers
        self.opts.layers = [int(x) for x in layers.split(",")]
        steps = self.opts.steps
        self.opts.steps = [int(x) for x in steps.split(",")]

        clases = self.opts.classes
        clases = list(set(clases.split(',')))#Class list
        clases = [c.lower() for c in clases]
        self.opts.classes = clases

        # --- set logging data
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir + "/" + self.opts.exp_name + ".log"

        # --- TODO: Move this create dir to check inputs function
        self._check_out_dir(self.opts.work_dir)
        self.opts.checkpoints = os.path.join(self.opts.work_dir, "checkpoints/")
        # if self.opts.do_class:
        #    self.opts.line_color = 1
        # --- define network output channels based on inputs

        return self.opts

    def __str__(self):
        """pretty print handle"""
        data = "------------ Options -------------"
        try:
            for k, v in sorted(vars(self.opts).items()):
                data = data + "\n" + "{0:15}\t{1}".format(k, v)
        except:
            data = data + "\nNo arguments parsed yet..."

        data = data + "\n---------- End  Options ----------\n"
        return data

    def __repr__(self):
        return self.__str__()
