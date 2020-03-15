# S8
Repository for Assignment S8
In this assignment Resnet architecture is used to train CIPHAR10 dataset. Developed APIs so as to load data, train, test and show results.

Following changes are done:
1. Dropout is added to the Base ResNet18 model
2. Added random cropping Image augmentation 
3. Added L2 regularization

Final Validation Accuracy : 85.46%
Number of Epochs: 30

Logs are as below:

  0%|          | 0/391 [00:00<?, ?it/s]

EPOCH: 0

/content/resnet18.py:65: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(out)
Loss=6.314135551452637 Batch_id=390 Accuracy=41.31: 100%|██████████| 391/391 [03:15<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.5755, Accuracy: 4248/10000 (42.48%)

EPOCH: 1

Loss=3.737288236618042 Batch_id=390 Accuracy=47.85: 100%|██████████| 391/391 [03:15<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.4401, Accuracy: 4662/10000 (46.62%)

EPOCH: 2

Loss=3.3232052326202393 Batch_id=390 Accuracy=50.54: 100%|██████████| 391/391 [03:15<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.8097, Accuracy: 4090/10000 (40.90%)

EPOCH: 3

Loss=2.9845097064971924 Batch_id=390 Accuracy=52.52: 100%|██████████| 391/391 [03:16<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.5037, Accuracy: 4755/10000 (47.55%)

EPOCH: 4

Loss=2.944131374359131 Batch_id=390 Accuracy=54.24: 100%|██████████| 391/391 [03:16<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.2624, Accuracy: 5426/10000 (54.26%)

EPOCH: 5

Loss=2.9348692893981934 Batch_id=390 Accuracy=55.80: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.4421, Accuracy: 4830/10000 (48.30%)

EPOCH: 6

Loss=1.7273719310760498 Batch_id=390 Accuracy=66.28: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.8376, Accuracy: 7082/10000 (70.82%)

EPOCH: 7

Loss=1.9250555038452148 Batch_id=390 Accuracy=68.54: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.8824, Accuracy: 6944/10000 (69.44%)

EPOCH: 8

Loss=1.5370872020721436 Batch_id=390 Accuracy=69.55: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.8735, Accuracy: 6924/10000 (69.24%)

EPOCH: 9

Loss=1.7402743101119995 Batch_id=390 Accuracy=70.75: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.7753, Accuracy: 7297/10000 (72.97%)

EPOCH: 10

Loss=1.8002607822418213 Batch_id=390 Accuracy=71.17: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.7797, Accuracy: 7301/10000 (73.01%)

EPOCH: 11

Loss=1.7638154029846191 Batch_id=390 Accuracy=72.04: 100%|██████████| 391/391 [03:16<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.7805, Accuracy: 7334/10000 (73.34%)

EPOCH: 12

Loss=1.3410749435424805 Batch_id=390 Accuracy=78.11: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5546, Accuracy: 8099/10000 (80.99%)

EPOCH: 13

Loss=1.2048401832580566 Batch_id=390 Accuracy=79.92: 100%|██████████| 391/391 [03:16<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5342, Accuracy: 8181/10000 (81.81%)

EPOCH: 14

Loss=1.0997071266174316 Batch_id=390 Accuracy=80.37: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5164, Accuracy: 8235/10000 (82.35%)

EPOCH: 15

Loss=1.2152645587921143 Batch_id=390 Accuracy=80.92: 100%|██████████| 391/391 [03:15<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5135, Accuracy: 8279/10000 (82.79%)

EPOCH: 16

Loss=1.2264512777328491 Batch_id=390 Accuracy=81.49: 100%|██████████| 391/391 [03:15<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5035, Accuracy: 8303/10000 (83.03%)

EPOCH: 17

Loss=1.1242256164550781 Batch_id=390 Accuracy=81.71: 100%|██████████| 391/391 [03:15<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4997, Accuracy: 8313/10000 (83.13%)

EPOCH: 18

Loss=1.0195937156677246 Batch_id=390 Accuracy=83.99: 100%|██████████| 391/391 [03:15<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4636, Accuracy: 8462/10000 (84.62%)

EPOCH: 19

Loss=1.1037931442260742 Batch_id=390 Accuracy=84.65: 100%|██████████| 391/391 [03:14<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4588, Accuracy: 8474/10000 (84.74%)

EPOCH: 20

Loss=0.965283989906311 Batch_id=390 Accuracy=84.84: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4579, Accuracy: 8465/10000 (84.65%)

EPOCH: 21

Loss=1.0757477283477783 Batch_id=390 Accuracy=85.29: 100%|██████████| 391/391 [03:16<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4544, Accuracy: 8479/10000 (84.79%)

EPOCH: 22

Loss=1.0101183652877808 Batch_id=390 Accuracy=85.36: 100%|██████████| 391/391 [03:15<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4539, Accuracy: 8483/10000 (84.83%)

EPOCH: 23

Loss=0.859192967414856 Batch_id=390 Accuracy=85.52: 100%|██████████| 391/391 [03:15<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4483, Accuracy: 8507/10000 (85.07%)

EPOCH: 24

Loss=0.9554926156997681 Batch_id=390 Accuracy=86.07: 100%|██████████| 391/391 [03:15<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4445, Accuracy: 8531/10000 (85.31%)

EPOCH: 25

Loss=0.7960834503173828 Batch_id=390 Accuracy=86.17: 100%|██████████| 391/391 [03:14<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4459, Accuracy: 8525/10000 (85.25%)

EPOCH: 26

Loss=0.9614863395690918 Batch_id=390 Accuracy=86.38: 100%|██████████| 391/391 [03:15<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4454, Accuracy: 8528/10000 (85.28%)

EPOCH: 27

Loss=0.8264625072479248 Batch_id=390 Accuracy=86.19: 100%|██████████| 391/391 [03:15<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4464, Accuracy: 8514/10000 (85.14%)

EPOCH: 28

Loss=1.0380147695541382 Batch_id=390 Accuracy=86.25: 100%|██████████| 391/391 [03:15<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4452, Accuracy: 8537/10000 (85.37%)

EPOCH: 29

Loss=1.1167728900909424 Batch_id=390 Accuracy=86.41: 100%|██████████| 391/391 [03:15<00:00,  2.48it/s]


Test set: Average loss: 0.4458, Accuracy: 8546/10000 (85.46%)

