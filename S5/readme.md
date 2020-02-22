Model1:
1.Targets:

Get the set-up right
Set Transforms
Set Data Loader
Set Basic Working Code
Set Basic Training & Test Loop

2.Results:
Parameters: 6,379,786
Best Training Accuracy: 99.98
Best Test Accuracy: 99.33

3.Analysis:
Extremely Heavy Model for such a problem
Model is over-fitting, but we are changing our model in the next step

 

4.File Link:

https://github.com/sudhakarmlal/EVA4/blob/master/S5/Model1.ipynb


Model2:


For your second attempt, please share your:

Targets:
Results: (must include best train/test accuracies and total parameters)
Analysis:
File Link:
Your answer:
 (Links to an external site.)

Targets:Make the model lighter
 (Links to an external site.)

 

 (Links to an external site.)

 2.Results: (must include best train/test accuracies and total parameters)

 (Links to an external site.)

Parameters: 9,990
Best Train Accuracy: 99.14
Best Test Accuracy: 98.59

 (Links to an external site.)

3.Analysis:

 (Links to an external site.)

Good model!
Not much over-fitting, model is capable if pushed further

 (Links to an external site.)

 

 (Links to an external site.)

4.File Link:

https://github.com/sudhakarmlal/EVA4/blob/master/S5/Model2.ipy




Model3:


1.Target:Include Batch Normalization

 

2.Results:

Parameters: 10,154
Best Train Accuracy: 99.84
Best Test Accuracy: 99.13

3.Analysis:

We have started to see over-fitting now.
Even if the model is pushed further, it won't be able to get to 99.4

4.File Link:

https://github.com/sudhakarmlal/EVA4/blob/master/S5/Model3.ipynb


Model4:

Your answer:
1.Target:Include Regularization through Dropout

 

2.Results:

Parameters: 10,154
Training accuracy:98.63
Test accuracy:99.10

3.Analysis:

Overfitting reduced to a larget extent.

But yet to reach the goal

4.File Link:

https://github.com/sudhakarmlal/EVA4/blob/master/S5/Model4.ipynb (Links to an external site.)

 
 
 
 Model5:
 
 Your answer:
1.Target: Reach 99.4 in  less than  15 epochs with   less than  10K parameters

To be considered:

Added image rotation
Changed BatchSize
Reduced the number of parameters
Decreased the dropout to  0.01

Gap layer added
 

 

2.Results:

Total params: 8,824
Training Accuracy:99.04
Test Accuracy:99.43

3.Analysis:

Overfitting is not there.

Changing the batch size to  32, reducing the no of 3*3    channels (for reducing the parameters) ,adding image rotation has really helped to improve the model performance

4.File Link:

https://github.com/sudhakarmlal/EVA4/blob/master/S5/Model5.ipynb (Links to an external site.)




