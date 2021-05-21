import numpy as np
import random
from tqdm import tqdm
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel_dir', required=True)
    parser.add_argument('--wav_dir', required=True)
    parser.add_argument('--train_dir', default='./dump/train_data')
    parser.add_argument('--val_dir', default='./dump/val_data')
    args = parser.parse_args()

    mel_dir = args.mel_dir
    wav_dir = args.wav_dir
    train_dump_dir = args.train_dir
    val_dump_dir = args.val_dir

    os.makedirs(train_dump_dir, exist_ok=True)
    os.makedirs(val_dump_dir, exist_ok=True)

    print("Prepare training data(95%) and validation data(5%)...")

    l = os.listdir(mel_dir)
    file_num = len(l)
    count = 0
    for p in tqdm(l):
        basename = p[4:-4]
        if count < file_num * 0.95:
            np.save(os.path.join(train_dump_dir, f'{basename}-feats.npy'), np.load(os.path.join(mel_dir, f'mel-{basename}.npy')), allow_pickle=False)
            np.save(os.path.join(train_dump_dir, f'{basename}-wave.npy'), np.load(os.path.join(wav_dir, f'audio-{basename}.npy')), allow_pickle=False)
        else:
            np.save(os.path.join(val_dump_dir, f'{basename}-feats.npy'), np.load(os.path.join(mel_dir, f'mel-{basename}.npy')), allow_pickle=False)
            np.save(os.path.join(val_dump_dir, f'{basename}-wave.npy'), np.load(os.path.join(wav_dir, f'audio-{basename}.npy')), allow_pickle=False)
        count += 1

    print("Task Done!")



