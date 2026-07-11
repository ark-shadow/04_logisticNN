import numpy as np
import copy
import os
from PIL import Image
from sklearn.model_selection import train_test_split

import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("preetviradiya/brian-tumor-dataset")

print("Path to dataset files:", path)
meta = pd.read_csv(os.path.join(path, "metadata.csv"))
print(meta.head())

tumor_dir = os.path.join(path, "Brain Tumor Data Set", "Brain Tumor Data Set", "Brain Tumor")
normal_dir = os.path.join(path, "Brain Tumor Data Set", "Brain Tumor Data Set", "Healthy")


metadata_path = os.path.join(path, "metadata.csv")
meta = pd.read_csv(metadata_path)[["class"]]
print("Metadata preview:\n", meta.head())


IMG_SIZE = 64  # resize to 128x128 maybe


def process_image(img_path):
    try:
        img = Image.open(img_path).convert("L")  # grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array.flatten()
    except Exception as e:
        print(f"Skipped {img_path}: {e}")
        return None

X, y = [], []

# Tumor images = label 1
for filename in os.listdir(tumor_dir):
    img_path = os.path.join(tumor_dir, filename)
    img_vector = process_image(img_path)
    if img_vector is not None:
        X.append(img_vector)
        y.append(1)

# Normal images = label 0
for filename in os.listdir(normal_dir):
    img_path = os.path.join(normal_dir, filename)
    img_vector = process_image(img_path)
    if img_vector is not None:
        X.append(img_vector)
        y.append(0)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)


print("Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


y_train = y_train.reshape(-1, 1)
y_val   = y_val.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

def initParams():
  W1 = np.random.randn(4096, 25) * np.sqrt(2. / 4096)
  b1 = np.zeros((25))

  W2 = np.random.randn(25, 15) * np.sqrt(2. / 25)
  b2 = np.zeros((15))

  W3 = np.random.randn(15, 1) * np.sqrt(2. / 15)
  b3 = np.zeros((1))

  return W1, b1, W2, b2, W3, b3

def sigmoid (z):
    return 1 / (1+ np.exp(-z))

def reLU(z):                 #return 0 for any negative value of z, if z > 0 return z
  return np.maximum(0, z)

def derivReLU(z):
  return (z > 0).astype(float)


def forwardProp(X, W1, b1, W2, b2, W3, b3):
  m, n = X.shape

  #Layer1 (25 neurons):  4096 inputs (pixels), 4096 weights per neuron, 25 neurons
  #(X_train, 4096) @ (4096, 25) = (X_train, 25)

  f_Wb1 = np.dot(X, W1) + b1
  A1 = reLU(f_Wb1)

  #Layer 2 (15 neurons):   25 inputs, 25 weights per input, 15 neurons
  #(X_train, 25) @ (25, 15) = (X_train, 15)

  f_Wb2 = np.dot(A1, W2) + b2
  A2 = reLU(f_Wb2)

  #Layer 3 (1 neuron): 15 inputs, 15 weights per input, 1 neurons
  #(X_train, 15) @ (15, 1) = (X_train, 1)

  f_Wb3 = np.dot(A2, W3) + b3

  A3 = sigmoid(f_Wb3)

  return A1, A2, A3, f_Wb1, f_Wb2

def computeCost (A3, y):
  m = y.shape[0]

  epsilon = 1e-8

  cost = -1/m * np.sum(y*np.log(A3+epsilon) + (1-y)*np.log(1-A3+epsilon))

  return cost

def backProp (X, y, A1, A2, A3, W1, W2, W3, z1, z2):
  #A1, A2, A3: activation values                     z1, z2 are pre-activation (no reLU)
  m = X.shape[0]

  dA3 = A3 - y          #yhat - y

  #Layer 3: output layer

  dj_dw3 = (1/m) * np.dot(A2.T, dA3) #gradient of cost with respect to weights in Layer 3:

  dj_db3 = (1/m) * np.sum(dA3, axis = 0) #gradient of cost with respect to bias in Layer 3:

  #Layer 2:

  dA2 = np.dot(dA3, W3.T) #calculate gradient flowing to previous layer:
  dZ2 = dA2 * derivReLU(z2)   #the derivative of activation before reLU

  dj_dw2 = (1/m) * np.dot(A1.T, dZ2) #gradient of cost with respect to weights in Layer 2
  dj_db2 = (1/m) * np.sum(dZ2, axis = 0)

  #Layer 1:

  dA1 = np.dot(dZ2, W2.T) #calculate gradient flowing to layer 1
  dZ1 = dA1 * derivReLU(z1)

  dj_dw1 = (1/m) * np.dot(X.T, dZ1) #gradient of cost with respect to input
  dj_db1 = (1/m) * np.sum(dZ1, axis = 0)

  return dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3

def getAccuracy(A3, y):
  predictions = (A3 >= 0.5).astype(int)

  accuracy = np.mean(predictions==y)

  return accuracy*100

def train(X, y, epochs, alpha):
  W1, b1, W2, b2, W3, b3 = initParams()

  for i in range (epochs):
     #Forward Propagation
    A1, A2, A3, z1, z2 = forwardProp(X, W1, b1, W2, b2, W3, b3)

    #Backward Propagation
    dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3 = backProp(X, y, A1, A2, A3, W1, W2, W3, z1, z2)


    #Update Parameters:
    W1 -= alpha * dj_dw1
    b1 -= alpha * dj_db1
    W2 -= alpha * dj_dw2
    b2 -= alpha * dj_db2
    W3 -= alpha * dj_dw3
    b3 -= alpha * dj_db3

    if (i+1)%20 == 0 or i == 0:
      cost = computeCost(A3, y)
      acc = getAccuracy(A3, y)
      print(f"Epoch {i+1}: Cost = {cost:.4f}, accuracy = {acc:.2f}%")

    if (i % 200 == 0):
      if cost < 0.15:
        alpha /= 1.25
      elif cost < 0.3:
        alpha /= 1.5


  return W1, b1, W2, b2, W3, b3


trainW1, trainb1, trainW2, trainb2, trainW3, trainb3 = train(X_train, y_train, 2000, 0.03)

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    # We only need the final output A3 from forwardProp
    _, _, A3, _, _ = forwardProp(X, W1, b1, W2, b2, W3, b3)
    predictions = (A3 >= 0.5).astype(int).flatten()
    return predictions

# Check Validation set
val_preds = make_predictions(X_val, trainW1, trainb1, trainW2, trainb2, trainW3, trainb3)
val_accuracy = np.mean(val_preds == y_val.flatten())
print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%")

# Test it on the test set
test_preds = make_predictions(X_test, trainW1, trainb1, trainW2, trainb2, trainW3, trainb3)
test_accuracy = np.mean(test_preds == y_test.flatten())
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")