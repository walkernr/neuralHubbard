# -*- coding: utf-8 -*-
"""
Created on Wed May 8 17:53:27 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelBinarizer
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from sklearn.model_selection import train_test_split
from scipy.odr import ODR, Model as ODRModel, RealData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-pt', '--plot', help='plot results', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-g', '--gpu', help='gpu run (will default to cpu if unable)', action='store_true')
    parser.add_argument('-ad', '--anomaly_detection', help='anomaly detection for embedding', action='store_true')
    parser.add_argument('-nt', '--threads', help='number of threads',
                        type=int, default=20)
    parser.add_argument('-n', '--name', help='simulation name',
                            type=str, default='hubbard')
    parser.add_argument('-t', '--times', help='time slice count',
                        type=int, default=100)
    parser.add_argument('-k', '--kpoints', help='k-point count',
                        type=int, default=16)
    parser.add_argument('-ui', '--unsuper_interval', help='interval for selecting phase points (manifold)',
                        type=int, default=1)
    parser.add_argument('-un', '--unsuper_samples', help='number of samples per phase point (manifold)',
                        type=int, default=500)
    parser.add_argument('-si', '--super_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1000)
    parser.add_argument('-sc', '--scaler', help='feature scaler',
                        type=str, default='minmax')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=8)
    parser.add_argument('-mf', '--manifold', help='manifold learning method',
                        type=str, default='tsne')
    parser.add_argument('-cl', '--clustering', help='clustering method',
                        type=str, default='dbscan')
    parser.add_argument('-nc', '--clusters', help='number of clusters (neighbor criterion eps for dbscan)',
                        type=float, default=1e-3)
    parser.add_argument('-bk', '--backend', help='keras backend',
                        type=str, default='tensorflow')
    parser.add_argument('-opt', '--optimizer', help='optimization function',
                        type=str, default='nadam')
    parser.add_argument('-lss', '--loss', help='loss function',
                        type=str, default='mse')
    parser.add_argument('-ep', '--epochs', help='number of epochs',
                        type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                        type=float, default=1e-3)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=256)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.anomaly_detection, args.threads, args.name, args.times, args.kpoints,
            args.unsuper_interval, args.unsuper_samples, args.super_interval, args.super_samples,
            args.scaler, args.latent_dimension, args.manifold, args.clustering,
            args.clusters, args.backend, args.optimizer, args.loss, args.epochs, args.learning_rate, args.random_seed)


def write_specs():
    if VERBOSE:
        print(100*'-')
        print('input summary')
        print(100*'-')
        print('plot:                      %d' % PLOT)
        print('parallel:                  %d' % PARALLEL)
        print('gpu:                       %d' % GPU)
        print('anomaly detection:         %d' % AD)
        print('threads:                   %d' % THREADS)
        print('name:                      %s' % NAME)
        print('time slices:               %d' % NT)
        print('k-points:                  %d' % NK)
        print('random seed:               %d' % SEED)
        print('unsuper interval:          %d' % UNI)
        print('unsuper samples:           %d' % UNS)
        print('super interval:            %d' % SNI)
        print('super samples:             %d' % SNS)
        print('scaler:                    %s' % SCLR)
        print('latent dimension:          %d' % LD)
        print('manifold learning:         %s' % MNFLD)
        print('clustering:                %s' % CLST)
        if CLST == 'dbscan':
            print('neighbor eps:              %.2e' % NC)
        else:
            print('clusters:                  %d' % NC)
        print('backend:                   %s' % BACKEND)
        print('network:                   %s' % 'cnn2d')
        print('optimizer:                 %s' % OPT)
        print('loss function:             %s' % LSS)
        print('epochs:                    %d' % EP)
        print('learning rate:             %.2e' % LR)
        print('fitting function:          %s' % 'logistic')
        print(100*'-')
    with open(OUTPREF+'.out', 'w') as out:
        out.write(100*'-' + '\n')
        out.write('input summary\n')
        out.write(100*'-' + '\n')
        out.write('plot:                      %d\n' % PLOT)
        out.write('parallel:                  %d\n' % PARALLEL)
        out.write('gpu:                       %d\n' % GPU)
        out.write('anomaly detection:         %d\n' % AD)
        out.write('threads:                   %d\n' % THREADS)
        out.write('name:                      %s\n' % NAME)
        out.write('time slices:               %d\n' % NT)
        out.write('k-points:                  %d\n' % NK)
        out.write('random seed:               %d\n' % SEED)
        out.write('unsuper interval:          %d\n' % UNI)
        out.write('unsuper samples:           %d\n' % UNS)
        out.write('super interval:            %d\n' % SNI)
        out.write('super samples:             %d\n' % SNS)
        out.write('scaler:                    %s\n' % SCLR)
        out.write('latent dimension:          %d\n' % LD)
        out.write('manifold learning:         %s\n' % MNFLD)
        out.write('clustering:                %s\n' % CLST)
        if CLST == 'dbscan':
            out.write('neighbor eps:              %.2e\n' % NC)
        else:
            out.write('clusters:                  %d\n' % NC)
        out.write('backend:                   %s\n' % BACKEND)
        out.write('network:                   %s\n' % 'cnn2d')
        out.write('optimizer:                 %s\n' % OPT)
        out.write('loss function:             %s\n' % LSS)
        out.write('epochs:                    %d\n' % EP)
        out.write('learning rate:             %.2e\n' % LR)
        out.write('fitting function:          %s\n' % 'logistic')
        out.write(100*'-' + '\n')


def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))


def absolute(beta, t):
    a, b, c, d = beta
    return a*np.power(np.abs(t-b), c)+d


def odr_fit(func, dom, mrng, srng, pg):
    ''' performs orthogonal distance regression '''
    dat = RealData(dom, mrng, EPS*np.ones(len(dom)), srng+EPS)
    mod = ODRModel(func)
    odr = ODR(dat, mod, pg)
    odr.set_job(fit_type=0)
    fit = odr.run()
    popt = fit.beta
    perr = fit.sd_beta
    ndom = 128
    fdom = np.linspace(np.min(dom), np.max(dom), ndom)
    fval = func(popt, fdom)
    return popt, perr, fdom, fval


def gauss_sampling(beta):
    z_mean, z_log_var = beta
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def build_variational_autoencoder():
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    # encoder layers
    input = Input(shape=(NT, NKS, NKS), name='encoder_input')
    conv0 = Conv3D(filters=32, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(input)
    conv1 = Conv3D(filters=64, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(2, 1, 1))(conv0)
    conv2 = Conv3D(filters=32, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(conv1)
    conv3 = Conv3D(filters=64, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(2, 1, 1))(conv2)
    shape = K.int_shape(conv3)
    fconv3 = Flatten()(conv3)
    d0 = Dense(1024, activation='relu')(fconv3)
    z_mean = Dense(LD, name='z_mean')(d0)
    z_log_var = Dense(LD, name='z_log_std')(d0) # more numerically stable to use log(var_z)
    z = Lambda(gauss_sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
    # construct encoder
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    if VERBOSE:
        print('encoder network summary')
        print(100*'-')
        encoder.summary()
        print(100*'-')
    # decoder layers
    latent_input = Input(shape=(LD,), name='z_sampling')
    d1 = Dense(np.prod(shape[1:]), activation='relu')(latent_input)
    rd1 = Reshape(shape[1:])(d1)
    convt0 = Conv3DTranspose(filters=64, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(2, 1, 1))(rd1)
    convt1 = Conv3DTranspose(filters=32, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(convt0)
    convt2 = Conv3DTranspose(filters=64, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(2, 1, 1))(convt1)
    convt3 = Conv3DTranspose(filters=32, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(convt2)
    output = Conv3DTranspose(filters=1, kernel_size=3, activation='sigmoid',
                             kernel_initializer='he_normal', padding='same', name='decoder_output')(convt3)
    # construct decoder
    decoder = Model(latent_input, output, name='decoder')
    if VERBOSE:
        print('decoder network summary')
        print(100*'-')
        decoder.summary()
        print(100*'-')
    # construct vae
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae_mlp')
    reconstruction_losses = {'bc': lambda a, b: binary_crossentropy(a, b),
                             'mse': lambda a, b: mse(a, b)}
    # vae loss
    reconstruction_loss = NT*reconstruction_losses[LSS](K.flatten(input), K.flatten(output))
    kl_loss = -0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss+kl_loss)
    vae.add_loss(vae_loss)
    # compile vae
    vae.compile(optimizer=OPTS[OPT])
    # return vae networks
    return encoder, decoder, vae


def random_selection(dat, intrvl, ns):
    rdat = dat[::intrvl, ::intrvl]
    ned, nbeta, _, _, _ = rdat.shape
    idat = np.zeros((ned, nbeta, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting random classification samples from full data')
        print(100*'-')
    for i in tqdm(range(ned), disable=not VERBOSE):
        for j in tqdm(range(nbeta), disable=not VERBOSE):
                idat[i, j] = np.random.permutation(rdat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldat = np.array([[rdat[i, j, idat[i, j], :, :] for j in range(nbeta)] for i in range(ned)])
    return sldat


def inlier_selection(dat, intrvl, ns):
    rdat = dat[::intrvl, ::intrvl]
    ned, nbeta, _, _, _ = rdat.shape
    if AD:
        lof = LocalOutlierFactor(contamination='auto', n_jobs=THREADS)
    idat = np.zeros((ned, nbeta, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting inlier samples from classification data')
        print(100*'-')
    for i in tqdm(range(ned), disable=not VERBOSE):
        for j in tqdm(range(nbeta), disable=not VERBOSE):
                if AD:
                    fpred = lof.fit_predict(rdat[i, j, :, 0])
                    try:
                        idat[i, j] = np.random.choice(np.where(fpred==1)[0], size=ns, replace=False)
                    except:
                        idat[i, j] = np.argsort(lof.negative_outlier_factor_)[:ns]
                else:
                    idat[i, j] = np.random.permutation(rdat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nbeta)] for i in range(ned)])
    return sldat


if __name__ == '__main__':
    # parse command line arguments
    (VERBOSE, PLOT, PARALLEL, GPU, AD, THREADS, NAME, NT, NK,
     UNI, UNS, SNI, SNS,
     SCLR, LD, MNFLD, CLST, NC,
     BACKEND, OPT, LSS, EP, LR, SEED) = parse_args()
    NKS = np.int32(np.sqrt(NK))
    if CLST == 'dbscan':
        NCS = '%.0e' % NC
    else:
        NC = int(NC)
        NCS = '%d' % NC
    CWD = os.getcwd()
    EPS = 0.025
    # number of phases
    NPH = 3
    # number of embedding dimensions
    ED = 2

    np.random.seed(SEED)
    # environment variables
    os.environ['KERAS_BACKEND'] = BACKEND
    if BACKEND == 'tensorflow':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow import set_random_seed
        set_random_seed(SEED)
    if PARALLEL:
        if not GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['MKL_NUM_THREADS'] = str(THREADS)
        os.environ['GOTO_NUM_THREADS'] = str(THREADS)
        os.environ['OMP_NUM_THREADS'] = str(THREADS)
        os.environ['openmp'] = 'True'
    else:
        THREADS = 1
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from keras.models import Model
    from keras.layers import Input, Lambda, Dense, Conv1D, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, Flatten, Reshape
    from Conv1DTranspose import Conv1DTranspose
    from keras.losses import binary_crossentropy, mse
    from keras.optimizers import SGD, Adadelta, Adam, Nadam
    from keras.callbacks import History, CSVLogger, ReduceLROnPlateau
    from keras import backend as K
    if PLOT:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        plt.rc('font', family='sans-serif')
        FTSZ = 28
        FIGW = 16
        PPARAMS = {'figure.figsize': (FIGW, FIGW),
                   'lines.linewidth': 4.0,
                   'legend.fontsize': FTSZ,
                   'axes.labelsize': FTSZ,
                   'axes.titlesize': FTSZ,
                   'axes.linewidth': 2.0,
                   'xtick.labelsize': FTSZ,
                   'xtick.major.size': 20,
                   'xtick.major.width': 2.0,
                   'ytick.labelsize': FTSZ,
                   'ytick.major.size': 20,
                   'ytick.major.width': 2.0,
                   'font.size': FTSZ}
        plt.rcParams.update(PPARAMS)
        SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
        CM = plt.get_cmap('plasma')

    OUTPREF = CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.%d.%d.%s.%s.%s.%d' % \
              (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, AD, MNFLD, CLST, NCS, SEED)
    write_specs()

    LDIR = os.listdir()

    try:
        CDAT = np.load(CWD+'/results/%s.%d.%d.%d.%d.%d.dat.c.npy' % (NAME, NT, NK, SNI, SNS, SEED))
        if VERBOSE:
            # print(100*'-')
            print('selected classification samples loaded from file')
            print(100*'-')
    except:
        DAT = np.load(CWD+'/results/%s.%d.%d.dat.npy' % (NAME, NT, NK))
        if VERBOSE:
            # print(100*'-')
            print('full dataset loaded from file')
            print(100*'-')
        CDAT = random_selection(DAT, SNI, SNS)
        del DAT
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%d.dat.c.npy' % (NAME, NT, NK, SNI, SNS, SEED), CDAT)
        if VERBOSE:
            print('selected classification samples generated')
            print(100*'-')
    CFL = np.load(CWD+'/results/%s.%d.%d.tfl.npy' % (NAME, NT, NK))[::SNI]
    CBETA = np.load(CWD+'/results/%s.%d.%d.tbeta.npy' % (NAME, NT, NK))[::SNI]
    SNFL, SNBETA = CFL.size, CBETA.size

    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler()}

    try:
        SCDAT = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%d.dmp.sc.npy' \
                        % (NAME, NT, NK, SNI, SNS, SCLR, SEED)).reshape(SNFL*SNBETA*SNS, NT, NKS, NKS)
        if VERBOSE:
            print('scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        CDAT = CDAT[:, :, :, :, :, np.newaxis]
        if SCLR == 'glbl':
            SCDAT = CDAT.reshape(SNFL*SNBETA*SNS, NT, NKS, NKS)
            for i in range(NKS):
                for j in range(NKS):
                    TMIN, TMAX = SCDAT[:, :, :, :, i, j].min(), SCDAT[:, :, :, :, i, j].max()
                    SCDAT[:, :, :, :, i, j] = (SCDAT[:, :, :, :, i, j]-TMIN)/(TMAX-TMIN)
            del TMIN, TMAX
        else:
            SCDAT = SCLRS[SCLR].fit_transform(CDAT.reshape(SNFL*SNBETA*SNS, NT*NK)).reshape(SNFL*SNBETA*SNS, NT, NKS, NKS)
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%d.dmp.sc.npy' % (NAME, NT, NK, SNI, SNS, SCLR, SEED), SCDAT.reshape(SNFL, SNBETA, SNS, NT, NKS, NKS))
        if VERBOSE:
            print('scaled selected classification samples computed')
            print(100*'-')

    OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=False),
            'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
            'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

    ENC, DEC, VAE = build_variational_autoencoder()

    try:
        VAE.load_weights(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.wt.h5' \
                         % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), by_name=True)
        TLOSS = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.loss.trn.npy' \
                        % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        VLOSS = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.loss.val.npy' \
                        % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        if VERBOSE:
            print('variational autoencoder trained weights loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('variational autoencoder training on scaled selected classification samples')
            print(100*'-')
        CSVLG = CSVLogger(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.log.csv'
                          % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), append=True, separator=',')
        LR_DECAY = ReduceLROnPlateau(monitor='val_loss', patience=8, verbose=VERBOSE)
        TRN, VAL = train_test_split(SCDAT, test_size=0.25, shuffle=True)
        VAE.fit(x=TRN, y=None, validation_data=(VAL, None), epochs=EP, batch_size=64,
                shuffle=True, verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
        del TRN, VAL
        TLOSS = VAE.history.history['loss']
        VLOSS = VAE.history.history['val_loss']
        VAE.save_weights(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.wt.h5'
                         % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.loss.trn.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), TLOSS)
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.vae.loss.val.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), VLOSS)
        if VERBOSE:
            print(100*'-')
            print('variational autoencoder weights trained')
            print(100*'-')

    if VERBOSE:
        print('variational autoencoder training history information')
        print(100*'-')
        print('| epoch | training loss | validation loss |')
        print(100*'-')
        for i in range(EP):
            print('%02d %.2f %.2f' % (i, TLOSS[i], VLOSS[i]))
        print(100*'-')

    with open(OUTPREF+'.out', 'a') as out:
        out.write('variational autoencoder training history information\n')
        out.write(100*'-' + '\n')
        out.write('| epoch | training loss | validation loss |\n')
        out.write(100*'-' + '\n')
        for i in range(EP):
            out.write('%02d %.2f %.2f\n' % (i, TLOSS[i], VLOSS[i]))
        out.write(100*'-' + '\n')

    try:
        ZENC = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.npy'
                       % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)).reshape(SNFL*SNBETA*SNS, 2, LD)
        ERR = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.err.npy'
                       % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        # del SCDAT
        if VERBOSE:
            print('z encodings of scaled selected classification samples loaded from file')
            print(100*'-')
            print('error: %f' % ERR)
            print(100*'-')
    except:
        if VERBOSE:
            print('predicting z encodings of scaled selected classification samples')
            print(100*'-')
        ZENC = np.array(ENC.predict(SCDAT, verbose=VERBOSE))
        ERR = np.mean(np.square(np.array(DEC.predict(ZENC[2, :, :], verbose=VERBOSE))-SCDAT))
        ZENC = np.swapaxes(ZENC, 0, 1)[:, :2, :]
        ZENC[:, 1, :] = np.exp(0.5*ZENC[:, 1, :])
        # del SCDAT
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), ZENC.reshape(SNFL, SNBETA, SNS, 2, LD))
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.err.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), ERR)
        if VERBOSE:
            print(100*'-')
            print('z encodings of scaled selected classification samples predicted')
            print(100*'-')
            print('error: %f' % ERR)
            print(100*'-')

    try:
        PZENC = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.pca.prj.npy'
                        % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)).reshape(SNFL*SNBETA*SNS, 2, LD)
        CZENC = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.pca.cmp.npy'
                        % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        VZENC = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.pca.var.npy'
                        % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        if VERBOSE:
            print('pca projections of z encodings  loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('pca projecting z encodings')
            print(100*'-')
        PCAZENC = PCA(n_components=LD)
        PZENC = np.zeros((SNFL*SNBETA*SNS, 2, LD))
        CZENC = np.zeros((2, LD, LD))
        VZENC = np.zeros((2, LD))
        for i in range(2):
            PZENC[:, i, :] = PCAZENC.fit_transform(ZENC[:, i, :])
            CZENC[i, :, :] = PCAZENC.components_
            VZENC[i, :] = PCAZENC.explained_variance_ratio_
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.pca.prj.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), PZENC.reshape(SNFL, SNBETA, SNS, 2, LD))
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.pca.cmp.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), CZENC)
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.zenc.pca.var.npy'
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), VZENC)

    if VERBOSE:
        print('pca fit information')
        print(100*'-')
        for i in range(2):
            if i == 0:
                print('mean z fit')
            if i == 1:
                print('stdv z fit')
            print(100*'-')
            print('components')
            print(100*'-')
            for j in range(LD):
                print(LD*'%f ' % tuple(CZENC[i, j, :]))
            print(100*'-')
            print('explained variances')
            print(100*'-')
            print(LD*'%f ' % tuple(VZENC[i, :]))
            print(100*'-')
    with open(OUTPREF+'.out', 'a') as out:
        out.write('pca fit information\n')
        out.write(100*'-'+'\n')
        for i in range(2):
            if i == 0:
                out.write('mean z fit\n')
            if i == 1:
                out.write('stdv z fit\n')
            out.write(100*'-'+'\n')
            out.write('principal components\n')
            out.write(100*'-'+'\n')
            for j in range(LD):
                out.write(LD*'%f ' % tuple(CZENC[i, j, :]) + '\n')
            out.write(100*'-'+'\n')
            out.write('explained variances\n')
            out.write(100*'-'+'\n')
            out.write(LD*'%f ' % tuple(VZENC[i, :]) + '\n')
            out.write(100*'-'+'\n')

    def vae_plots():
        outpref = CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d' % \
                  (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.scatter(ZENC[:, 0, 0], ZENC[:, 1, 0],
                   cmap=plt.get_cmap('plasma'),
                   s=64, alpha=0.5, edgecolors='')
        plt.xlabel('mu')
        plt.ylabel('sigma')
        fig.savefig(outpref+'.vae.prj.ld.png')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.scatter(PZENC[:, 0, 0], PZENC[:, 1, 0],
                   cmap=plt.get_cmap('plasma'),
                   s=64, alpha=0.5, edgecolors='')
        plt.xlabel('mu')
        plt.ylabel('sigma')
        fig.savefig(outpref+'.vae.pca.prj.ld.png')
        plt.close()

        DIAGMLV = SCLRS['minmax'].fit_transform(np.mean(ZENC.reshape(SNFL, SNBETA, SNS, 2*LD), 2).reshape(SNFL*SNBETA, 2*LD)).reshape(SNFL, SNBETA, 2, LD)
        DIAGSLV = SCLRS['minmax'].fit_transform(np.var(ZENC.reshape(SNFL, SNBETA, SNS, 2*LD)/CBETA[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(SNFL*SNBETA, 2*LD)).reshape(SNFL, SNBETA, 2, LD)

        DIAGMPLV = SCLRS['minmax'].fit_transform(np.mean(PZENC.reshape(SNFL, SNBETA, SNS, 2*LD), 2).reshape(SNFL*SNBETA, 2*LD)).reshape(SNFL, SNBETA, 2, LD)
        for i in range(LD):
            if DIAGMPLV[0, 0, 0, i] > DIAGMPLV[-1, 0, 0, i]:
                DIAGMPLV[:, :, 0, i] = 1-DIAGMPLV[:, :, 0, i]
            if DIAGMPLV[int(SNFL/2), 0, 1, i] > DIAGMPLV[int(SNFL/2), -1, 1, i]:
                DIAGMPLV[:, :, 1, i] = 1-DIAGMPLV[:, :, 1, i]
        DIAGSPLV = SCLRS['minmax'].fit_transform(np.var(PZENC.reshape(SNFL, SNBETA, SNS, 2*LD)/CBETA[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(SNFL*SNBETA, 2*LD)).reshape(SNFL, SNBETA, 2, LD)

        for i in range(2):
            for j in range(2):
                for k in range(LD):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        ax.imshow(DIAGMLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    if i == 1:
                        ax.imshow(DIAGSLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(CBETA.size), minor=True)
                    ax.set_yticks(np.arange(CFL.size), minor=True)
                    plt.xticks(np.arange(CBETA.size)[::1], np.round(CBETA, 2)[::1], rotation=-60)
                    plt.yticks(np.arange(CFL.size)[::1], np.round(CFL, 2)[::1])
                    plt.xlabel('BETA')
                    plt.ylabel('CHEMICAL POTENTIAL')
                    fig.savefig(outpref+'.vae.diag.ld.%d.%d.%d.png' % (i, j, k))
                    plt.close()
        for i in range(2):
            for j in range(2):
                for k in range(LD):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        ax.imshow(DIAGMPLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    if i == 1:
                        ax.imshow(DIAGSPLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(CBETA.size), minor=True)
                    ax.set_yticks(np.arange(CFL.size), minor=True)
                    plt.xticks(np.arange(CBETA.size)[::4], np.round(CBETA, 2)[::4], rotation=-60)
                    plt.yticks(np.arange(CFL.size)[::4], np.round(CFL, 2)[::4])
                    plt.xlabel('BETA')
                    plt.ylabel('CHEMICAL POTENTIAL')
                    fig.savefig(outpref+'.vae.diag.ld.pca.%d.%d.%d.png' % (i, j, k))
                    plt.close()

    if PLOT:
        vae_plots()

    try:
        SLPZENC = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.%d.%d.%d.zenc.pca.prj.inl.npy' \
                          % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, AD, SEED))
        # del PZENC, CZENC, VZENC, CDAT
        if VERBOSE:
            print('inlier selected z encodings loaded from file')
            print(100*'-')
    except:
        SLPZENC = inlier_selection(PZENC.reshape(SNFL, SNBETA, SNS, 2, LD), UNI, UNS)
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.%d.%d.%d.zenc.pca.prj.inl.npy' \
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, AD, SEED), SLPZENC)
        # del PZENC, CZENC, VZENC, CDAT
        if VERBOSE:
            print('inlier selected z encodings computed')
            print(100*'-')

    UFL, UBETA = CFL[::UNI], CBETA[::UNI]
    UNFL, UNBETA = UFL.size, UBETA.size

    # reduction dictionary
    MNFLDS = {'pca':PCA(n_components=2),
              'kpca':KernelPCA(n_components=2, n_jobs=THREADS),
              'isomap':Isomap(n_components=2, n_jobs=THREADS),
              'lle':LocallyLinearEmbedding(n_components=2, n_jobs=THREADS),
              'tsne':TSNE(n_components=2, perplexity=UNS,
                          early_exaggeration=24, learning_rate=200, n_iter=1000,
                          verbose=VERBOSE, n_jobs=THREADS)}

    try:
        MSLZENC = np.load(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.%d.%s.%d.%d.zenc.mfld.inl.npy' \
                          % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, MNFLD, AD, SEED))
        if VERBOSE:
            print('inlier selected z encoding manifold loaded from file')
            print(100*'-')
    except:
        MSLZENC = np.zeros((UNFL*UNBETA*UNS, 2, 2))
        for i in range(ED):
            MSLZENC[:, i, :] = MNFLDS[MNFLD].fit_transform(SLPZENC[:, :, :, i, :].reshape(UNFL*UNBETA*UNS, LD))
        np.save(CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.%d.%s.%d.%d.zenc.mfld.inl.npy' \
                % (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, MNFLD, AD, SEED), MSLZENC)
        if VERBOSE:
            if MNFLD == 'tsne':
                print(100*'-')
            print('inlier selected z encoding manifold computed')
            print(100*'-')

    if PLOT:
        outpref = CWD+'/results/%s.%d.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d.%d.%d.%s.%d' % \
                  (NAME, NT, NK, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, AD, MNFLD, SEED)
        for i in range(2):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.scatter(MSLZENC[:, i, 0], MSLZENC[:, i, 1],
                       cmap=plt.get_cmap('plasma'),
                       s=64, alpha=0.5, edgecolors='')
            plt.xlabel('mu')
            plt.ylabel('sigma')
            fig.savefig(OUTPREF+'.vae.mnfld.prj.ld.%02d.png' % i)