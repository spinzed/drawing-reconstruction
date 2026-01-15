# Variational Encoder for Handwritten image reconstruction

## Table of Contents

1. Variational Encoder for Handwritten image reconstruction  
2. Upute za instalaciju  
3. Struktura repozitorija  
   - models  
   - data  
   - train  
4. Training

---

## Upute za instalaciju

`quickdraw/download.sh` da potegne dataset.  
Za sada samo jedan file, kasnije ćemo ostale kada sredimo training loop.

`scripts/train.py` je training entrypoint.  
Kada popravimo train loop trebalo bi dodati i skriptu za generiranje.

---

## Struktura repozitorija

### models

Contains out conditional and classical VAE implementation.

1. **vae_model.py**  
   Classical Vae model using the same loss described in lecture.  
   It has encoder and decoder, decoder is using Normal distribution forcing Normal distribution on latent variable.  
   Decoder uses bernoulli distribution as it better models generation of binary images.

2. **cvae_model.py**  
   Version of conditional VAE that uses two encoders (`encoder_prior` and `decoder_post`) and one decoder.  
   - `encoder_prior` models `p(z|y)`  
   - `encoder_post` models `p(z|y, x)`  
   - decoder models `p(x|z, y)`  

   Reminder:  
   - `y` is partial image  
   - `x` is full image  

   This naming convention is not being respected in train scripts!!!

   Encoders consist of 4 layers of Convolutional layers and a linear layer at the end.  
   Decoder consist of 4 layers of ConvTranspose layers to upsample latent vector.

3. **cvae_model_decoderz.py**  
   This version is almost the same as `cvae_model.py`.  
   Difference is that decoder does not get partial image and only gets `z`.

   This is done because its been noted that model can overfit to partial image instead of trying to reconstruct it.

---

### data

1. **dataset.py**  
   Contains implementations of torch dataset and dataloader class for vae models.

   Dataloader returns data in this order:

(partial image, full image, label)


Inside dataloader class image erosion is used to widen drawings.

2. **utils.py**  
Has functions that transform sequential stroke data from quick draw to image specified by `image_shape` parameter.

---

### train

1. **train_cvae.py**  
Can be used to train both `cvae_model.py` and `cvae_model_decoderz.py`, only thing you need to do is modify imports.

During training canvas visualizes training.  
At the end of training validation and train losses are shown.  
Model is saved on `KeyboardInterrupt` and it can save checkpoints.

You can easily configure:
1. learning rate using `lr`
2. latent dimension using `latent_dim`
3. binarization threshold (it seems to be it is little biased toward 1, so i use `0.55`)
4. image_limit which says how many images will be used in training
5. image_size which is used as parameter to dataset class

Keep in mind that for now X and y variables from dataloaders are switched in naming convention.

2. **train_vae.py**  
Same parameters but is used to train `vae_model_bernoulli.py`.

---

### Google colab notebooks


## Training

To train cvae model just run `train_cvae.py` script.
