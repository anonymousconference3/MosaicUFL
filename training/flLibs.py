# Standard libs
import os, re, shutil, sys, time, datetime, logging, pickle, json, socket
import random, math, gc, copy
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import multiprocessing, threading
import numpy as np
from collections import deque
from collections import OrderedDict
import collections
import numba

import torch.nn as nn
from transformers import AlbertTokenizer, AlbertForMaskedLM,AlbertConfig, AlbertForSequenceClassification
# PyTorch libs
import torch
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels
from torch.utils.data.sampler import WeightedRandomSampler
#  from torch_baidu_ctc import CTCLoss

# libs from FLBench
from argParser import args
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner,select_index_dataset
#from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model
from transformers import AutoTokenizer, AutoModelForMaskedLM

nlp_model_path ='/home/zikang/code/albert-base-v2'
if args.task == 'nlp':
    from utils.nlp import *
elif args.task == 'speech':
    from utils.speech import SPEECH
    from utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
    from utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
    from utils.speech import BackgroundNoiseDataset

from helper.clientSampler import clientSampler
from utils.yogi import YoGi

# shared functions of aggregator and clients
# initiate for nlp
tokenizer = None

if args.task == 'nlp' or args.task == 'text_clf':
    tokenizer = AlbertTokenizer.from_pretrained(nlp_model_path)
    #tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    #tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

modelDir = os.path.join(args.log_path, args.model)
modelPath = modelDir+'/'+str(args.model)+'.pth.tar' if args.model_path is None else args.model_path
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out
    

def init_dataset():
    global tokenizer

    outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                }

    logging.info("====Initialize the model")
    if args.load_model:
        try:
            with open(args.load_path, 'rb') as fin:
                model = pickle.load(fin)

            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            sys.exit(-1)
    else:
        if args.task == 'nlp':
            # we should train from scratch
            #config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
            #model = AutoModelWithLMHead.from_config(config)
            model = AlbertForMaskedLM.from_pretrained(nlp_model_path)
        elif args.task == 'text_clf':
            config = AlbertConfig()

            # Set the number of labels for the sequence classification task
            config.num_labels = outputClass[args.data_set]

            # Load the pre-trained model weights along with the modified config
            model = AlbertForSequenceClassification.from_pretrained(nlp_model_path, num_labels=5)

        elif args.task == 'tag-one-sample':
            # Load LR model for tag prediction
            model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
        elif args.task == 'speech':
            if args.model == 'mobilenet':
                from utils.resnet_speech import mobilenet_v2
                model = mobilenet_v2(num_classes=outputClass[args.data_set], inchannels=1)
            elif args.model == "resnet18":
                from utils.resnet_speech import resnet18
                model = resnet18(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet34":
                from utils.resnet_speech import resnet34
                model = resnet34(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet50":
                from utils.resnet_speech import resnet50
                model = resnet50(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet101":
                from utils.resnet_speech import resnet101
                model = resnet101(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet152":
                from utils.resnet_speech import resnet152
                model = resnet152(num_classes=outputClass[args.data_set], in_channels=1)
            else:
                # Should not reach here
                logging.info('Model must be resnet or mobilenet')
                sys.exit(-1)

        elif args.task == 'voice':
            from utils.voice_model import DeepSpeech, supported_rnns

            # Initialise new model training
            with open(args.labels_path) as label_file:
                labels = json.load(label_file)

            audio_conf = dict(sample_rate=args.sample_rate,
                                window_size=args.window_size,
                                window_stride=args.window_stride,
                                window=args.window,
                                noise_dir=args.noise_dir,
                                noise_prob=args.noise_prob,
                                noise_levels=(args.noise_min, args.noise_max))
            model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                                nb_layers=args.hidden_layers,
                                labels=labels,
                                rnn_type=supported_rnns[args.rnn_type.lower()],
                                audio_conf=audio_conf,
                                bidirectional=args.bidirectional)
        else:
            if args.model == 'resnet9':
                from utils.models import resnet9
                model = resnet9()
            elif args.model == 'resnet18':
                from utils.resnet_speech import ResNet18
                model = ResNet18(num_classes=outputClass[args.data_set])
            elif args.data_set == "cifar10":
                model = get_model('fedfixmatch', "Cifar")
            else:
                logging.info("open_img model here")
                model = get_model()
    
    

    train_dataset, test_dataset = [], []

    # Load data if the machine acts as clients
    if args.this_rank != 0 or args.semi == True:

        if args.data_set == 'Mnist':
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                          transform=test_transform)

        elif args.data_set == 'cifar10':
            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                            transform=test_transform)

        elif args.data_set == "imagenet":
            train_transform, test_transform = get_data_transform('imagenet')
            train_dataset = datasets.ImageNet(args.data_dir, split='train', download=False, transform=train_transform)
            test_dataset = datasets.ImageNet(args.data_dir, split='val', download=False, transform=test_transform)

        elif args.data_set == 'emnist':
            test_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
            train_dataset = datasets.EMNIST(args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

        elif args.data_set == 'femnist':
            from utils.femnist import FEMNIST

            train_transform, test_transform = get_data_transform('mnist')
            train_dataset = FEMNIST(args.data_dir, train=True, transform=train_transform)
            test_dataset = FEMNIST(args.data_dir, train=False, transform=test_transform)

        elif args.data_set == 'openImg':
            from utils.openImg import OpenImage

            transformer_ns = 'openImg' if args.model != 'inception_v3' else 'openImgInception'
            train_transform, test_transform = get_data_transform(transformer_ns)
            train_dataset = OpenImage(args.data_dir, dataset='train', transform=train_transform)
            test_dataset = OpenImage(args.data_dir, dataset='test', transform=test_transform)

        elif args.data_set == 'blog':
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
            test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        elif args.data_set == 'stackoverflow':
            from utils.stackoverflow import stackoverflow

            train_dataset = stackoverflow(args.data_dir, train=True)
            test_dataset = stackoverflow(args.data_dir, train=False)

        elif args.data_set == 'yelp':
            import utils.dataloaders as fl_loader

            train_dataset = fl_loader.TextSentimentDataset(args.data_dir, train=True, tokenizer=tokenizer, max_len=args.clf_block_size)
            test_dataset = fl_loader.TextSentimentDataset(args.data_dir, train=False, tokenizer=tokenizer, max_len=args.clf_block_size)

        elif args.data_set == 'google_speech':
            bkg = '_background_noise_'
            data_aug_transform = transforms.Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
            bg_dataset = BackgroundNoiseDataset(os.path.join(args.data_dir, bkg), data_aug_transform)
            add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
            train_dataset = SPEECH(args.data_dir, train= True,
                                    transform=transforms.Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))
            valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
            test_dataset = SPEECH(args.data_dir, train=False,
                                    transform=transforms.Compose([LoadAudio(),
                                             FixAudioLength(),
                                             valid_feature_transform]))
        elif args.data_set == 'common_voice':
            from utils.voice_data_loader import SpectrogramDataset
            train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                           manifest_filepath=args.train_manifest,
                                           labels=model.labels,
                                           normalize=True,
                                           speed_volume_perturb=args.speed_volume_perturb,
                                           spec_augment=args.spec_augment,
                                           data_mapfile=args.data_mapfile)
            test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                          manifest_filepath=args.test_manifest,
                                          labels=model.labels,
                                          normalize=True,
                                          speed_volume_perturb=False,
                                          spec_augment=False)
        else:
            print('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog', 'stackoverflow', 'speech', 'yelp']))
            sys.exit(-1)

    #model = get_model('fedfixmatch', "Cifar")
    return model, train_dataset, test_dataset


def get_model(name = None, backbone = None):
    if args.load_model:
        try:
            with open(args.load_path, 'rb') as fin:
                model = pickle.load(fin)

            logging.info("====Load model successfully\n")
            return model
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            sys.exit(-1)

    outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                }
    
    if name == 'local':
        return
    elif name == 'global':
        return
    elif name == 'fedfixmatch' and backbone == 'Mnist':
        print("using resnet 9 for mnist")
        model = MnistNet().to('cuda')
    elif name == 'fedfixmatch' and backbone == 'Cifar':
        print("using resnet 9 for Cifar10")
        model = Net().to('cuda')
        #model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])
    elif name == 'fedfixmatch' and backbone == 'Cifar100':
        print("using resnet 9 for Cifar100")
        model = Cifar100Net().to('cuda')
    elif name == 'fedfixmatch' and backbone == 'Svhn':
        print("using resnet 9 for SVHN")
        model = Net().to('cuda')
    else:
        if args.task == 'nlp':
            # we should train from scratch
            # config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
            # model = AutoModelWithLMHead.from_config(config)
            model = AlbertForMaskedLM.from_pretrained(nlp_model_path)

        elif args.task == 'text_clf':
            config = AlbertConfig()

            # Set the number of labels for the sequence classification task
            config.num_labels = outputClass[args.data_set]

            # Load the pre-trained model weights along with the modified config
            model = AlbertForSequenceClassification.from_pretrained(nlp_model_path, num_labels=5)


        elif args.task == 'tag-one-sample':
            # Load LR model for tag prediction
            model = LogisticRegression(args.vocab_token_size, args.vocab_tag_size)
        elif args.task == 'speech':
            if args.model == 'mobilenet':
                from utils.resnet_speech import mobilenet_v2
                model = mobilenet_v2(num_classes=outputClass[args.data_set], inchannels=1)
            elif args.model == "resnet18":
                from utils.resnet_speech import resnet18
                model = resnet18(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet34":
                from utils.resnet_speech import resnet34
                model = resnet34(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet50":
                from utils.resnet_speech import resnet50
                model = resnet50(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet101":
                from utils.resnet_speech import resnet101
                model = resnet101(num_classes=outputClass[args.data_set], in_channels=1)
            elif args.model == "resnet152":
                from utils.resnet_speech import resnet152
                model = resnet152(num_classes=outputClass[args.data_set], in_channels=1)
            else:
                # Should not reach here
                logging.info('Model must be resnet or mobilenet')
                sys.exit(-1)

        elif args.task == 'voice':
            from utils.voice_model import DeepSpeech, supported_rnns

            # Initialise new model training
            with open(args.labels_path) as label_file:
                labels = json.load(label_file)

            audio_conf = dict(sample_rate=args.sample_rate,
                                window_size=args.window_size,
                                window_stride=args.window_stride,
                                window=args.window,
                                noise_dir=args.noise_dir,
                                noise_prob=args.noise_prob,
                                noise_levels=(args.noise_min, args.noise_max))
            model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                                nb_layers=args.hidden_layers,
                                labels=labels,
                                rnn_type=supported_rnns[args.rnn_type.lower()],
                                audio_conf=audio_conf,
                                bidirectional=args.bidirectional)
        else:
            if args.model == 'resnet9':
                from utils.models import resnet9
                model = resnet9()
            elif args.model == 'resnet18':
                from utils.resnet_speech import ResNet18
                model = ResNet18(num_classes=outputClass[args.data_set])
            else:
                logging.info("model hhhhhere")
                model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])
    return model






