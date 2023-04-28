 PassGAN

This repository contains code for the [_PassGAN: A Deep Learning Approach for Password Guessing_](https://arxiv.org/abs/1709.00440) paper. 

The model from PassGAN is taken from [_Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028) and it is assumed that the authors of PassGAN used the [improved_wgan_training](https://github.com/igul222/improved_wgan_training) tensorflow implementation in their work. For this reason, I have modified that reference implementation in this repository to make it easy to train (`train.py`) and sample (`sample.py`) from. This repo contributes:

- A command-line interface
- I have made several improvements to conserve enough free memory during the loading and filtering of wordlists. 
- Additionally, I have implemented various fixes to ensure compatibility with recent TensorFlow versions. 
- Furthermore, I have removed unused code from the original transfer learning implementation.
- A pretrained PassGAN model trained on the RockYou dataset
- A pretrained PassGAN model trained on the darkc0de+openwall+xato-net-10M dataset 11 chars
- A pretrained PassGAN model trained on the crackstation-only-human dataset 10 chars


## Getting Started

```bash
# requires python3 CUDA **8+** to be pre-installed **works on latest versions of cuda & 2.5.1 tensorflow**
pip install -r requirements.txt
```

### Generating password samples

Use the pretrained model to generate 1,000,000 passwords, saving them to `gen_passwords.txt`.

[This model is based on rockyou and is the original one that comes with paper and brannondorsey] (https://github.com/brannondorsey/PassGAN)  version
```bash
python sample.py \
	--input-dir pretrained \
	--checkpoint pretrained/checkpoints/195000.ckpt \
	--output gen_passwords.txt \
	--batch-size 1024 \
	--num-samples 1000000
```

Use the commonu11 model to generate 10,000,000 passwords, saving them to `gen_passwords.txt`.
[This model is based on darkc0de+openwall+xato-net-10M filtered and unique](https://github.com/danielmiessler/SecLists/tree/master/Passwords) 11 chars sequence
```bash
python sample.py \
	--input-dir commonu11 \
	--checkpoint commonu11/checkpoints/checkpoint_195000.ckpt \
	--output gen_passwords.txt \
	--seq-length 11 \
	--batch-size 1024 \
	--num-samples 10000000
```

Use the csho10 model to generate 100,000,000 passwords, saving them to `gen_passwords.txt`.
[This model is trained on crackstation-only-human.txt dictionary](https://download.g0tmi1k.com/wordlists/large/crackstation-human-only.txt.gz) 10 chars sequence
```bash
python sample.py \
	--input-dir csho10 \
	--checkpoint csho10/checkpoints/checkpoint_150000.ckpt \
	--output gen_passwords.txt \
	--batch-size 1024 \
	--num-samples 100000000
```

For enhanced cracking performance, you can combine the original wordlist with the generated one to create unique passwords. This is because the model tends to repeat some words in large generations and does not include words from the original list.

The default value for --seq-length is 10. In the commonu11 case, the seq-length should always be set to 11, which corresponds to the training character length. If you train your own dataset with a longer length, you must adjust the seq-length to match the length used during training when using sample.py.


### Training your own models

Training a model on a large dataset (100MB+) can take several hours on a GTX 1080. (9+h)

```bash
# download the rockyou training data
# contains 80% of the full rockyou passwords (with repeats)
# that are 10 characters or less
curl -L -o data/train.txt https://github.com/brannondorsey/PassGAN/releases/download/data/rockyou-train.txt

# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python train.py --output-dir output --training-data data/train.txt
```

You are encouraged to train using your own password leaks and datasets. Some great places to find those include:

[BitTorrent Rocktastic:](https://labs.nettitude.com/torrents/Rocktastic12a.rar.torrent)
**13 GB and 1,133,849,621 words**

Make sure you filter by length of password to save some memory

## The basic attack plan to cracking hashes would be as follows:

- Make custom personalized wordlist and make custom straight attack
- Apply rules to previous wordlist
- Straight attack on common passwords (wordlists from leaks)
- Apply rules to previous wordlist
- Here comes PassGAN mixed with original wordlist and unified. You can replace previous step with this one too
- You can continue generating PassGAN lists with straight attack 
- -or- you can continue to bruteforce -or- apply masks with combinations not already checked

## Attribution and License

This code is released under an [MIT License](https://github.com/igul222/improved_wgan_training/blob/master/LICENSE). You are free to use, modify, distribute, or sell it under those terms. 

The majority of the credit for the code in this repository goes to @igul222 for his work on the [improved_wgan_training](https://github.com/igul222/improved_wgan_training). I've simply modularized his code a bit, added a command-line interface, and specialized it for the PassGAN paper.

The PassGAN [research and paper](https://arxiv.org/abs/1709.00440) was published by Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, Fernando Perez-Cruz.
