import argparse

from p05b_lwr import main as p05b


p05b(tau=5e-1,train_path='../data/ds5_train.csv',eval_path='../data/ds5_valid.csv')


