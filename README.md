
# Table of Contents

1.  [Variational Encoder for Handwritten image reconstruction](#org3ea5ef6)
    1.  [Upute za instalaciju](#orgb6098a1)
    2.  [Modeli](#orga873cf3)
        1.  [VAE sa linearnim slojevima i MSE gubitkom](#org9b6a69f)
        2.  [VAE sa konvolucijskim slojevima i VAE gubitkom](#org4811315)



<a id="org3ea5ef6"></a>

# Variational Encoder for Handwritten image reconstruction


<a id="orgb6098a1"></a>

## Upute za instalaciju

quickdraw/download.sh da potegne dataset. Za sada samo jedan file, kasnije ćemo ostale kada sredimo training loop.
scripts/train.py je training entrypoint. Kada popravimo train loop trebalo bi dodati i skriptu za generiranje.
\_


<a id="orga873cf3"></a>

## Modeli


<a id="org9b6a69f"></a>

### VAE sa linearnim slojevima i MSE gubitkom

train.py
vae\_model.py


<a id="org4811315"></a>

### VAE sa konvolucijskim slojevima i VAE gubitkom

train\_vae.py
vae\_model\_classic.py

