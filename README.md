# DENet: Disentangled Embedding Network for Visible Watermark Removal

This is official implementation of paper DENet: Disentangled Embedding Network for Visible Watermark Removal [[AAAI2023 Oral](https://ojs.aaai.org/index.php/AAAI/article/view/25337)]

<img src='imgs/framework.png'>

## Dataset preparation
~~~
|--data
|--|--LOGO
   |--|--10kmid
   |--|--10kgray
   |--|--10khigh
~~~
## Installation
```bash
pip install -r requirements.txt
```
## Training
### Train on LOGO-H
```bash
bash scripts/train_contrast_attention_on_logo_high.sh 
```
### Train on LOGO-L
```bash
bash scripts/train_contrast_attention_on_logo_mid.sh 
```
### Train on LOGO-Gray
```bash
bash scripts/train_contrast_attention_on_logo_gray.sh
```

## Testing
### Test on LOGO-H
```bash
bash scripts/test_LOGO_10khigh.sh
```
### Test on LOGO-L
```bash
bash scripts/test_LOGO_10kmid.sh
```
### Test on LOGO-Gray
```bash
bash scripts/test_LOGO_10kgray.sh
```

## Acknowledgements
This code is mainly based on the previous work [SLBR](https://github.com/bcmi/SLBR-Visible-Watermark-Removal)