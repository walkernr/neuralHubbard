# -*- coding: utf-8 -*-
"""
Created on Wed May 8 11:13:55 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='simulation name',
                    type=str, default='hubbard')
parser.add_argument('-t', '--times', help='time slice count',
                    type=int, default=100)
parser.add_argument('-k', '--kpoints', help='k-point count',
                    type=int, default=16)
parser.add_argument('-cd', '--convolution_dimension', help='dimensionality of convolutions (1d or 2d)',
                    type=int, default=3)
args = parser.parse_args()

NAME = args.name
NT = args.times
NK = args.kpoints
CD = args.convolution_dimension

NBETA = 25
NS = 10000

CWD = os.getcwd()
DATDIR = CWD+'/data'
SFL = ['F90']
NFL = len(SFL)
DATFLS = [[DATDIR+'/'+SFL[i]+'/'+f for f in os.listdir(DATDIR+'/'+SFL[i]) if '.dat' in f] for i in range(NFL)]

if CD == 3:
    NKS = 4
    ORDER = np.array([1, 13, 14, 15, 9, 10, 11, 12, 5, 6, 7, 8, 2, 3, 4, 16], dtype=np.int32)-1


def read_file(fn):
    tfl, beta = np.array(re.findall(r'\d+', fn.split('/')[-1])).astype(np.float)
    dat = []
    n = 0
    with open(fn, 'r') as f:
        it = iter(f)
        for l in it:
            if 'config' in l:
                n += 1
                d = ''
                while True:
                    dl = it.readline()
                    if dl.strip() == '' or 'cpu  ,time=' in dl:
                        break
                    else:
                        d += dl
                dat.append(d.strip().split())
    if CD == 1:
        dat = np.array(dat).astype(np.float32).reshape(n, NT, NK)
    if CD == 3:
        dat = np.array(dat).astype(np.float32).reshape(n, NT, NK)[:, :, ORDER].reshape(n, NT, NKS, NKS)
    return dat, tfl/1e4, beta

FL = np.zeros((NFL, NBETA), dtype=np.float32)
BETA = np.zeros((NFL, NBETA), dtype=np.float32)
if CD == 1:
    DAT = np.zeros((NFL, NBETA, NS, NT, NK), dtype=np.float)
if CD == 3:
    DAT = np.zeros((NFL, NBETA, NS, NT, NKS, NKS), dtype=np.float)
print('parsing data files')
for i in tqdm(range(NFL)):
    for j in tqdm(range(NBETA)):
        DAT[i, j], FL[i, j], BETA[i, j] = read_file(DATFLS[i][j])
TFL = np.array([SFL[i][1:] for i in range(NFL)]).astype(np.float32)/1e2
TBETA = BETA[0]

np.save(CWD+'/results/%s.%d.%d.%d.tfl.npy' % (NAME, NT, NK, CD), TFL)
np.save(CWD+'/results/%s.%d.%d.%d.tbeta.npy' % (NAME, NT, NK, CD), TBETA)
np.save(CWD+'/results/%s.%d.%d.%d.fl.npy' % (NAME, NT, NK, CD), FL)
np.save(CWD+'/results/%s.%d.%d.%d.beta.npy' % (NAME, NT, NK, CD), BETA)
np.save(CWD+'/results/%s.%d.%d.%d.dat.npy' % (NAME, NT, NK, CD), DAT)