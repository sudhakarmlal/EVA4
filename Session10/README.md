# S10
Repository for S10 Use LRFinder and ReduceLRonPlateau using RestNet Model on Ciphar10 dataset

Assignment S10

In this assignment Resnet architecture is used to train CIPHAR10 dataset and used Data Augmmentation. Developed APIs so as to load data, train, test and show results. Best model is stored after each epoch
Following changes are done:

    Added Data Augmentations (Albumentatons) i. Cutout ii. Horizontal Flip iii. Gaussian Noise iv. Elastic Transform

    Added L2 regularization
    Added LRFinder to get good starting learning rate
    Used ReduceLROnPlateau strategey for scheduling learning rates

Got best test accuracy: 88.80% (Epoch 37). Best model after each epoch is stored and retrieved

Logs are as below:

   0%|          | 0/196 [00:00<?, ?it/s]

EPOCH: 0

/content/models/resnet18.py:67: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(out)
Loss=1.4788159132003784 Batch_id=195 Accuracy=39.54: 100%|██████████| 196/196 [00:29<00:00,  6.73it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 1.6649, Accuracy: 4495/10000 (44.95%)

Test Accuracy: 44.95 has increased. Saving the model
EPOCH: 1

Loss=1.0059988498687744 Batch_id=195 Accuracy=54.84: 100%|██████████| 196/196 [00:29<00:00,  6.70it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 1.0163, Accuracy: 6565/10000 (65.65%)

Test Accuracy: 65.65 has increased. Saving the model
EPOCH: 2

Loss=0.9434942007064819 Batch_id=195 Accuracy=63.86: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.8111, Accuracy: 7214/10000 (72.14%)

Test Accuracy: 72.14 has increased. Saving the model
EPOCH: 3

Loss=0.923568844795227 Batch_id=195 Accuracy=69.28: 100%|██████████| 196/196 [00:29<00:00,  6.72it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 1.0465, Accuracy: 6706/10000 (67.06%)

EPOCH: 4

Loss=0.8866066932678223 Batch_id=195 Accuracy=72.95: 100%|██████████| 196/196 [00:29<00:00,  6.74it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 1.0379, Accuracy: 6715/10000 (67.15%)

EPOCH: 5

Loss=0.5896663069725037 Batch_id=195 Accuracy=75.39: 100%|██████████| 196/196 [00:28<00:00,  6.76it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.5823, Accuracy: 8026/10000 (80.26%)

Test Accuracy: 80.26 has increased. Saving the model
EPOCH: 6

Loss=0.7886285781860352 Batch_id=195 Accuracy=77.28: 100%|██████████| 196/196 [00:29<00:00,  6.68it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6921, Accuracy: 7844/10000 (78.44%)

EPOCH: 7

Loss=0.49160757660865784 Batch_id=195 Accuracy=79.46: 100%|██████████| 196/196 [00:29<00:00,  6.70it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.7367, Accuracy: 7770/10000 (77.70%)

EPOCH: 8

Loss=0.41679033637046814 Batch_id=195 Accuracy=80.39: 100%|██████████| 196/196 [00:28<00:00,  6.78it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6337, Accuracy: 8110/10000 (81.10%)

Test Accuracy: 81.1 has increased. Saving the model
EPOCH: 9

Loss=0.4310689866542816 Batch_id=195 Accuracy=82.07: 100%|██████████| 196/196 [00:29<00:00,  6.72it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6773, Accuracy: 8000/10000 (80.00%)

EPOCH: 10

Loss=0.4291706085205078 Batch_id=195 Accuracy=82.83: 100%|██████████| 196/196 [00:29<00:00,  6.75it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6610, Accuracy: 8175/10000 (81.75%)

Test Accuracy: 81.75 has increased. Saving the model
EPOCH: 11

Loss=0.4297536313533783 Batch_id=195 Accuracy=83.55: 100%|██████████| 196/196 [00:29<00:00,  6.71it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.7651, Accuracy: 7886/10000 (78.86%)

EPOCH: 12

Loss=0.2370925396680832 Batch_id=195 Accuracy=84.56: 100%|██████████| 196/196 [00:29<00:00,  6.70it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.7416, Accuracy: 8058/10000 (80.58%)

EPOCH: 13

Loss=0.4211389124393463 Batch_id=195 Accuracy=84.82: 100%|██████████| 196/196 [00:29<00:00,  6.72it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6440, Accuracy: 8228/10000 (82.28%)

Test Accuracy: 82.28 has increased. Saving the model
EPOCH: 14

Loss=0.4728224277496338 Batch_id=195 Accuracy=85.45: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6005, Accuracy: 8441/10000 (84.41%)

Test Accuracy: 84.41 has increased. Saving the model
EPOCH: 15

Loss=0.2555420398712158 Batch_id=195 Accuracy=85.74: 100%|██████████| 196/196 [00:29<00:00,  6.66it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6036, Accuracy: 8357/10000 (83.57%)

EPOCH: 16

Loss=0.4195984899997711 Batch_id=195 Accuracy=85.76: 100%|██████████| 196/196 [00:29<00:00,  6.71it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.5954, Accuracy: 8396/10000 (83.96%)

EPOCH: 17

Loss=0.38863784074783325 Batch_id=195 Accuracy=86.67: 100%|██████████| 196/196 [00:29<00:00,  6.75it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.7381, Accuracy: 8244/10000 (82.44%)

EPOCH: 18

Loss=0.40623655915260315 Batch_id=195 Accuracy=86.68: 100%|██████████| 196/196 [00:29<00:00,  6.70it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.6882, Accuracy: 8253/10000 (82.53%)

Epoch    19: reducing learning rate of group 0 to 4.0000e-03.
EPOCH: 19

Loss=0.1965150684118271 Batch_id=195 Accuracy=88.11: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4501, Accuracy: 8747/10000 (87.47%)

Test Accuracy: 87.47 has increased. Saving the model
EPOCH: 20

Loss=0.44211071729660034 Batch_id=195 Accuracy=88.47: 100%|██████████| 196/196 [00:29<00:00,  6.62it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4495, Accuracy: 8793/10000 (87.93%)

Test Accuracy: 87.93 has increased. Saving the model
EPOCH: 21

Loss=0.2983151972293854 Batch_id=195 Accuracy=88.99: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4383, Accuracy: 8802/10000 (88.02%)

Test Accuracy: 88.02 has increased. Saving the model
EPOCH: 22

Loss=0.23217546939849854 Batch_id=195 Accuracy=89.02: 100%|██████████| 196/196 [00:29<00:00,  6.66it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4408, Accuracy: 8817/10000 (88.17%)

Test Accuracy: 88.17 has increased. Saving the model
EPOCH: 23

Loss=0.4313240051269531 Batch_id=195 Accuracy=89.16: 100%|██████████| 196/196 [00:29<00:00,  6.70it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4448, Accuracy: 8805/10000 (88.05%)

EPOCH: 24

Loss=0.33717089891433716 Batch_id=195 Accuracy=89.05: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4323, Accuracy: 8824/10000 (88.24%)

Test Accuracy: 88.24 has increased. Saving the model
EPOCH: 25

Loss=0.19674794375896454 Batch_id=195 Accuracy=88.89: 100%|██████████| 196/196 [00:29<00:00,  6.62it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4366, Accuracy: 8828/10000 (88.28%)

Test Accuracy: 88.28 has increased. Saving the model
EPOCH: 26

Loss=0.23717176914215088 Batch_id=195 Accuracy=88.96: 100%|██████████| 196/196 [00:29<00:00,  6.63it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4368, Accuracy: 8829/10000 (88.29%)

Test Accuracy: 88.29 has increased. Saving the model
EPOCH: 27

Loss=0.2917460799217224 Batch_id=195 Accuracy=89.18: 100%|██████████| 196/196 [00:29<00:00,  6.67it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4399, Accuracy: 8846/10000 (88.46%)

Test Accuracy: 88.46 has increased. Saving the model
EPOCH: 28

Loss=0.3458470404148102 Batch_id=195 Accuracy=89.25: 100%|██████████| 196/196 [00:29<00:00,  6.68it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4389, Accuracy: 8858/10000 (88.58%)

Test Accuracy: 88.58 has increased. Saving the model
EPOCH: 29

Loss=0.30439019203186035 Batch_id=195 Accuracy=89.46: 100%|██████████| 196/196 [00:29<00:00,  6.71it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4418, Accuracy: 8857/10000 (88.57%)

EPOCH: 30

Loss=0.19815799593925476 Batch_id=195 Accuracy=89.21: 100%|██████████| 196/196 [00:29<00:00,  6.70it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4360, Accuracy: 8850/10000 (88.50%)

EPOCH: 31

Loss=0.1862778216600418 Batch_id=195 Accuracy=89.16: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4446, Accuracy: 8826/10000 (88.26%)

EPOCH: 32

Loss=0.2850312292575836 Batch_id=195 Accuracy=89.37: 100%|██████████| 196/196 [00:29<00:00,  6.69it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4420, Accuracy: 8857/10000 (88.57%)

Epoch    33: reducing learning rate of group 0 to 4.0000e-04.
EPOCH: 33

Loss=0.3567495346069336 Batch_id=195 Accuracy=89.42: 100%|██████████| 196/196 [00:29<00:00,  6.56it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4428, Accuracy: 8875/10000 (88.75%)

Test Accuracy: 88.75 has increased. Saving the model
EPOCH: 34

Loss=0.13625261187553406 Batch_id=195 Accuracy=89.39: 100%|██████████| 196/196 [00:29<00:00,  6.55it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4395, Accuracy: 8870/10000 (88.70%)

EPOCH: 35

Loss=0.38882458209991455 Batch_id=195 Accuracy=89.59: 100%|██████████| 196/196 [00:29<00:00,  6.57it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4414, Accuracy: 8868/10000 (88.68%)

EPOCH: 36

Loss=0.19582843780517578 Batch_id=195 Accuracy=89.31: 100%|██████████| 196/196 [00:30<00:00,  6.49it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4369, Accuracy: 8862/10000 (88.62%)

EPOCH: 37

Loss=0.36512327194213867 Batch_id=195 Accuracy=89.16: 100%|██████████| 196/196 [00:29<00:00,  6.56it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4417, Accuracy: 8880/10000 (88.80%)

Test Accuracy: 88.8 has increased. Saving the model
EPOCH: 38

Loss=0.43281954526901245 Batch_id=195 Accuracy=89.44: 100%|██████████| 196/196 [00:29<00:00,  6.59it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4404, Accuracy: 8867/10000 (88.67%)

EPOCH: 39

Loss=0.3416702449321747 Batch_id=195 Accuracy=89.46: 100%|██████████| 196/196 [00:30<00:00,  6.52it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4390, Accuracy: 8876/10000 (88.76%)

EPOCH: 40

Loss=0.27326861023902893 Batch_id=195 Accuracy=89.44: 100%|██████████| 196/196 [00:29<00:00,  6.56it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4425, Accuracy: 8867/10000 (88.67%)

EPOCH: 41

Loss=0.3372931480407715 Batch_id=195 Accuracy=89.48: 100%|██████████| 196/196 [00:29<00:00,  6.57it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4371, Accuracy: 8861/10000 (88.61%)

Epoch    42: reducing learning rate of group 0 to 4.0000e-05.
EPOCH: 42

Loss=0.33143019676208496 Batch_id=195 Accuracy=89.52: 100%|██████████| 196/196 [00:29<00:00,  6.62it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4439, Accuracy: 8876/10000 (88.76%)

EPOCH: 43

Loss=0.25506672263145447 Batch_id=195 Accuracy=89.45: 100%|██████████| 196/196 [00:29<00:00,  6.54it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4402, Accuracy: 8878/10000 (88.78%)

EPOCH: 44

Loss=0.24825496971607208 Batch_id=195 Accuracy=89.33: 100%|██████████| 196/196 [00:29<00:00,  6.61it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4382, Accuracy: 8871/10000 (88.71%)

EPOCH: 45

Loss=0.2984887957572937 Batch_id=195 Accuracy=89.64: 100%|██████████| 196/196 [00:29<00:00,  6.57it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4374, Accuracy: 8866/10000 (88.66%)

Epoch    46: reducing learning rate of group 0 to 4.0000e-06.
EPOCH: 46

Loss=0.3533063530921936 Batch_id=195 Accuracy=89.45: 100%|██████████| 196/196 [00:29<00:00,  6.58it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4403, Accuracy: 8874/10000 (88.74%)

EPOCH: 47

Loss=0.47478699684143066 Batch_id=195 Accuracy=89.46: 100%|██████████| 196/196 [00:29<00:00,  6.60it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4435, Accuracy: 8869/10000 (88.69%)

EPOCH: 48

Loss=0.36235785484313965 Batch_id=195 Accuracy=89.69: 100%|██████████| 196/196 [00:29<00:00,  6.66it/s]
  0%|          | 0/196 [00:00<?, ?it/s]


Test set: Average loss: 0.4386, Accuracy: 8875/10000 (88.75%)

EPOCH: 49

Loss=0.20658883452415466 Batch_id=195 Accuracy=89.41: 100%|██████████| 196/196 [00:29<00:00,  6.63it/s]


Test set: Average loss: 0.4391, Accuracy: 8875/10000 (88.75%)

Epoch    50: reducing learning rate of group 0 to 4.0000e-07.
