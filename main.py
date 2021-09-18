import pickle
from argparse import SUPPRESS, ArgumentParser, Namespace
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any, Callable, NoReturn, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from src.datamodules import DATA_MODULE_REGISTRY
from src.extractors import FeatureExtractor
from src.losses import LOSS_REGISTRY
from src.models import MODEL_REGISTRY
from src.reducers import REDUCER_REGISTRY
from src.utils import Null, int_or_str, preprocess_batch, to_numpy

torch.backends.cudnn.deterministic = True


def step(processor: Union[Callable, pl.LightningModule], dataloader: DataLoader,
         device: str = 'cuda', array_type: str = 'torch', epochs: int = 1,
         batch_size: int = 8, num_workers: int = 2,
         disable_progress_bar: bool = False) -> DataLoader:
    aggregated_X = []
    aggregated_y = []
    aggregated_ids = []
    for epoch in range(epochs):
        with tqdm(dataloader, disable=disable_progress_bar) as pbar:
            for i, data in enumerate(pbar):
                pbar.set_description(f'Epoch {epoch+1}/{epochs}')
                batch = preprocess_batch(data, array_type=array_type, device=device)
                inputs = batch['input']
                labels = batch['label']
                ids = batch['id']
                # HACK
                if hasattr(processor, 'encode'):
                    feats = processor.encode(inputs)
                else:
                    feats = processor(inputs)
                aggregated_X.append(to_numpy(feats))
                aggregated_y.append(to_numpy(labels))
                aggregated_ids.append(to_numpy(ids))

    X = torch.from_numpy(np.concatenate(aggregated_X))
    y = torch.from_numpy(np.concatenate(aggregated_y))
    ids = torch.from_numpy(np.concatenate(aggregated_ids))

    dataset = TensorDataset(X, y, ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    return dataloader


class LoaderCacheQuery():
    def __init__(self, args, cache_dir: PathLike = (Path.cwd() / 'cache')):
        self.cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        self.name_train = self.get_name(args, is_train=True)
        self.name_test = self.get_name(args, is_train=False)
        self.path_train_pkl = self.cache_dir / (self.name_train + '_train.pkl')
        self.path_test_pkl = self.cache_dir / (self.name_test + '_test.pkl')

    def set(self, train_loader: DataLoader, test_loader: DataLoader) -> NoReturn:
        with open(self.path_train_pkl, mode='wb') as f:
            pickle.dump(train_loader, f)
        with open(self.path_test_pkl, mode='wb') as f:
            pickle.dump(test_loader, f)

    def get(self) -> Tuple[Any, Any, bool]:
        if self.path_train_pkl.exists():
            with open(self.path_train_pkl, mode='rb') as f:
                train_loader = pickle.load(f)
            with open(self.path_test_pkl, mode='rb') as f:
                test_loader = pickle.load(f)
            return train_loader, test_loader, True
        else:
            return None, None, False

    @staticmethod
    def get_name(args: Namespace, is_train=True) -> str:
        # HACK
        if hasattr(args, 'without_background'):
            img_type = 'wo' if args.without_background else 'org'
        elif hasattr(args, 'class_name'):
            img_type = args.class_name
        elif hasattr(args, 'one_class'):
            img_type = args.one_class
        else:
            img_type = 'org'
        eblocks = 'eb' + ','.join([str(i) for i in args.extract_blocks])
        epochs = f'epoch{args.epochs}' if is_train else f'tta{args.tta}'
        name = f'{args.dataset_name}_{epochs}_{args.img_size}' + \
            f'_{args.fe_name}_{eblocks}_{args.pooling}_{args.seed}_{img_type}'
        return name


def extract_feature(args, feature_extractor, train_loader, test_loader) -> Tuple[DataLoader, DataLoader]:
    if args.use_cached_feature:
        lcq = LoaderCacheQuery(args)
        _train_loader, _test_loader, get_cached = lcq.get()
    else:
        lcq = None
        get_cached = False

    if get_cached:
        print('Successfully retrieved cached data.')
    else:
        _train_loader = step(feature_extractor, train_loader,
                             epochs=args.epochs, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             disable_progress_bar=args.disable_progress_bar)
        _test_loader = step(feature_extractor, test_loader,
                            epochs=args.tta, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            disable_progress_bar=args.disable_progress_bar)
        if lcq is not None:
            lcq.set(_train_loader, _test_loader)

    return _train_loader, _test_loader  # type: ignore


def reduce_feature(args, reducer, train_loader, test_loader) -> Tuple[DataLoader, DataLoader]:
    _data_type = reducer.data_type
    _train_loader = step(reducer, train_loader, epochs=1, array_type=_data_type,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         disable_progress_bar=args.disable_progress_bar)
    _test_loader = step(reducer, test_loader, epochs=1, array_type=_data_type,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        disable_progress_bar=args.disable_progress_bar)

    return _train_loader, _test_loader


def main(args, train_loader, test_loader, Reducer=Null,
         SimilarityEstimator=Null, pl_args=Namespace()) -> NoReturn:
    # ----------------------
    # 0. Seed Everything
    # ----------------------
    pl.seed_everything(args.seed)

    # ----------------------
    # 1. Feature Extraction
    # ----------------------
    print("[Feature Extraction] processing ...")
    feature_extractor = FeatureExtractor(
        model_name=args.fe_name, extract_blocks=args.extract_blocks, pooling=args.pooling, device=args.device)
    train_loader, test_loader = extract_feature(args, feature_extractor, train_loader, test_loader)
    del feature_extractor
    garbage_collection_cuda()

    # ----------------------------
    # 2. Dimensionality Reduction
    # ----------------------------
    if Reducer is not Null:
        print("[Dimensionality Reduction] training ...")
        if Reducer.run_type() == 'sklearn':
            reducer = Reducer.from_argparse_args(args)
            reducer.fit(deepcopy(train_loader))

        elif Reducer.run_type() == 'lightning':
            _input_dim = train_loader.dataset[:][0].shape[1]
            reducer = Reducer.from_argparse_args(args, input_dim=_input_dim)
            trainer = pl.Trainer(
                max_epochs=args.reducer_train_epochs, logger=None, checkpoint_callback=False,
                deterministic=True, gpus=args.gpus)
            trainer.fit(reducer, deepcopy(train_loader))
            if torch.cuda.is_available():
                reducer = reducer.cuda()

        else:
            raise TypeError("reducer's run type expected to be 'sklearn' or 'lightning'")

        print("[Dimensionality Reduction] processing ...")
        train_loader, test_loader = reduce_feature(args, reducer, train_loader, test_loader)

        del reducer
        garbage_collection_cuda()

    feat_dim = train_loader.dataset[:][0].shape[1]
    wandb.config.update({'feat_dim': feat_dim})
    print(f'Dimensionality of feature: {feat_dim}')

    # -------------------------
    # 3. Similarity estimation
    # -------------------------
    # loss
    loss = LOSS_REGISTRY(args.estimator_loss)
    # similarity estimator
    model = SimilarityEstimator.from_argparse_args(
        args, input_dim=feat_dim, loss=loss)

    # train & eval
    print("[Density Estimation] training ...")
    if model.run_type == 'lightning':
        trainer = pl.Trainer.from_argparse_args(
            pl_args, max_epochs=args.estimator_epochs, logger=WandbLogger(),
            checkpoint_callback=False, deterministic=True)
        trainer.fit(model, train_loader)
        print("[Density Estimation] evaluate ...")
        trainer.test(model, test_loader)

    elif model.run_type == 'sklearn':
        model.fit(train_loader)
        print("[Density Estimation] evaluate ...")
        result = model.test(test_loader)
        wandb.log(result)

    else:
        msg = "Simlarity estimator run type " + model.run_type + " not supported"
        raise TypeError(msg)

    del model
    garbage_collection_cuda()


def new_parser() -> ArgumentParser:
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root_path', type=str, default=SUPPRESS,
                        help="path where dataset is stored")
    parser.add_argument('--dataset_name', type=str,
                        choices=['mvtec', 'shanghai', 'cifar10'], default='mvtec',
                        help="name of anomaly deteciton dataset")
    parser.add_argument('--config_path', type=str, default='configs/mvtec.yaml',
                        help="filepath for a default experiment configuration")
    parser.add_argument('--img_size', type=int_or_str, default=SUPPRESS,
                        help="image size given for resizing")
    parser.add_argument('--fe_name', type=str, choices=['resnet18', 'resnet50-fractal', 'efficientnet-b5'], default=SUPPRESS,
                        help="name of the feature extractor (the pre-trained network)")
    parser.add_argument('--extract_blocks', type=lambda s: [int(i) for i in s.split(',')], default=SUPPRESS,
                        help="layers of the feature extractor to use")
    parser.add_argument('--pooling', type=str, default='avg',
                        choices=['min', 'max', 'avg', 'minmax', 'minavg', 'maxavg', 'minmaxavg'],
                        help="pooling type of feature extractor")
    parser.add_argument('--reducer_name', type=str, choices=['pca', 'ae', 'vae', 'none'], default='none',
                        help="name of dimensionality reduction methods")
    parser.add_argument('--estimator_name', type=str, choices=['flow', 'mvg', 'knn', 'ocsvm'], default=SUPPRESS,
                        help="name of the similarity estimator as a classifier for anomaly detection")
    parser.add_argument('--estimator_loss', type=str, choices=['nf_loss', 'none'], default=SUPPRESS,
                        help="name of loss function for similarity estimator")
    parser.add_argument('--estimator_epochs', type=int, default=1,
                        help="number of epochs for training similarity estimator")
    parser.add_argument('--epochs', type=int, default=SUPPRESS,
                        help="number of epochs for data augmentation")
    parser.add_argument('--batch_size', type=int, default=SUPPRESS,
                        help="size of the batches")
    parser.add_argument('--num_workers', type=int, default=SUPPRESS,
                        help="number of CPU workers")
    parser.add_argument('--use_cached_feature', action='store_true', default=False,
                        help='use cached feature, because loading images takes cost')
    parser.add_argument('--tta', type=int, default=SUPPRESS,
                        help="numbers of epochs for test-time augmentation")
    parser.add_argument('--seed', type=int, default=SUPPRESS,
                        help="seed for initializing training")
    parser.add_argument('--disable_progress_bar', action='store_true', default=SUPPRESS,
                        help="hide progress bar")

    return parser


if __name__ == '__main__':
    parser = new_parser()
    _base_args, _unknown = parser.parse_known_args()

    _default_conf = OmegaConf.load(_base_args.config_path)

    def merge_conf(args: ArgumentParser,
                   conf: DictConfig = _default_conf) -> DictConfig:
        # priority: args > defaults from yaml
        arg_conf = OmegaConf.create(vars(deepcopy(args)))
        conf = OmegaConf.merge(_default_conf, arg_conf)

        # post process
        def convert_none(x):
            return None if (isinstance(x, str) and x.lower() == 'none') else x
        conf.img_size = convert_none(conf.img_size)
        return conf

    base_args = merge_conf(_base_args)

    # add datamodule args
    MyDataModule = DATA_MODULE_REGISTRY(base_args.dataset_name)
    parser = MyDataModule.add_argparse_args(parser)
    # add reducer args
    MyReducer = REDUCER_REGISTRY(base_args.reducer_name)
    parser = MyReducer.add_argparse_args(parser)
    # add model args
    MyModel = MODEL_REGISTRY(base_args.estimator_name)
    parser = MyModel.add_argparse_args(parser)
    # parse args
    _args, _unknown = parser.parse_known_args()
    args = Namespace(**OmegaConf.to_container(merge_conf(_args)))

    # parse the rest to pl.Trainer
    pl_parser = pl.Trainer.add_argparse_args(ArgumentParser(allow_abbrev=False))
    pl_args = pl_parser.parse_args(_unknown)

    # create dataloaders
    dm = MyDataModule.from_argparse_args(args)
    train_loader, test_loader = dm.train_dataloader(), dm.test_dataloader()

    wandb.init(project='transfer-anomaly-detection', config=args)
    main(args, train_loader, test_loader,
         Reducer=MyReducer, SimilarityEstimator=MyModel, pl_args=pl_args)
