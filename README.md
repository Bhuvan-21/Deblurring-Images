# Deblurring-Images
In this project, I have trained a pix2pix like model on 100x100 images to increase the image quality.
The dataset link can be found in the train notebooks.

Models
Generator is the generator with 4 convolutional, 2 residual and 4 transpose conv layers.
In Generator_R, I used 3 conv, 4 residual and 3 t_conv layers.

Rest details can be found in models.py

Loss function

For Discriminator, the loss was Binary Cross Entropy only.

For Generator, the loss had
BCE for Disc output
Pixel wise loss, mean squared error : torch.mean((high-out)**2)
Standard deviation
Below I explain the reason,

BCE and Pixel Wise loss are intuitive and commonly used, to match output images.
Standard Deviation term, I used this because I was facing a problem in training, all pixels values were collapsing to 0. 

Note: I used the Perceptual loss function to match hd with output. The results were not better so I removed that term from the loss function.

The checkpoints I have stored are from, 22 epochs in G model and 7 epochs in GR model.

Face Detection

For this task, I have used opencv haarcascades and given a 10% margin as mentioned.
Currently I am only doing this for a single face but it can be easily extended into all the faces found using a loop.

Results (Validation Data)

![Alt text](Results/01.png?raw=true "Title")

