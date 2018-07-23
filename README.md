# minusGAN
this code is based on [ALICE](https://github.com/ChunyuanLI/ALICE).

### Dependencies
- `python 2.7 or 3.5`
- `TensorFlow 1.0.0`

we have two datasets X and Y,

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/X_train.png' width=250 />

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/Y_train.png'  width=250/>

plot them together in a pciture,

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/X_Y_train.png" align="left" width=250/>

and we wan to generate a dataset which is from the dataset X but not from the dataset Y, like the set operation X minus Y.

What minusGAN generate is

<img src='https://github.com/mathcbc/minusGAN/blob/master/results/minusGAN_result.png" align="left" width=250/>

