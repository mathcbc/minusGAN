# minusGAN
This code was based on [ALICE](https://github.com/ChunyuanLI/ALICE).

### Dependencies
- `python 2.7 or 3.5`
- `TensorFlow 1.0.0`

### Training the model
```bash
python minusGAN.py
```

### Experiment results on toy data
We have two datasets X and Y,

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/X_train.png' align="left" width=250 />
<img src='https://github.com/mathcbc/minusGAN/blob/master/results/Y_train.png'  width=250/> 

and plot them together in a pciture.

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/X_Y_train.png' width=250/>

We want to generate a dataset which is from the dataset X but not from the dataset Y, like the set operation of X minus Y.

What **minusGAN** generated are

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/minusGAN_result.png' align="left" width=250/>

