#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
import cv2
import numpy as np
import os

# Function to load and preprocess images
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # Assuming images are of fixed size, you may need to resize them
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())  # Flatten the image into a 1D array
            labels.append(int(filename.split('_')[0]))  # Assuming the label is in the filename
    return np.array(images), np.array(labels)

# Path to your training and testing datasets
train_path = r"C:\Users\Haier\Desktop\Aqsa\Training images"
test_path = r"C:\Users\Haier\Desktop\Aqsa\Testing images"

# Load and preprocess training images
X_train, y_train = load_images(train_path)

# Load and preprocess testing images
X_test, y_test = load_images(test_path)

# Create a Support Vector Machine classifier
classifier = svm.SVC(kernel='linear', C=1)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[2]:


print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# In[3]:


print("Loaded an example image:")
print(X_train[0])  # Assuming X_train is not empty


# In[7]:


print("SVM Classifier Parameters:")
print(classifier.get_params())  # Print classifier parameters


# In[8]:


import matplotlib.pyplot as plt

# Visualize an example image
plt.imshow(X_train[0].reshape((64, 64, 3)))  # Assuming images are color (3 channels)
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Visualize an example image
plt.imshow(X_train[1].reshape((64, 64, 3)))  # Assuming images are color (3 channels)
plt.show()


# In[10]:


import matplotlib.pyplot as plt

# ... (your existing code)

# Visualize a few example images with labels
num_images_to_visualize = 1500  # You can change this number as needed

for i in range(num_images_to_visualize):
    # Display the image
    plt.imshow(X_train[i].reshape((64, 64, 3)))  # Assuming images are color (3 channels)
    
    # Set the title with the corresponding label
    label = y_train[i]
    if label == 0:
        disease = "Diabetic Retinopathy"
    elif label == 1:
        disease = "Cataract"
    elif label == 2:
        disease = "Glaucoma"
    else:
        disease = "Unknown Label"
    
    plt.title(f"Label: {label} ({disease})")
    
    # Show the plot
    plt.show()


# In[13]:


import matplotlib.pyplot as plt

# ... (your existing code)

# Visualize a few example images with labels
num_images_to_visualize = 15

for i in range(num_images_to_visualize):
    plt.imshow(X_train[i].reshape((64, 64, 3)))  # Assuming images are color (3 channels)
    
    # Set the title with the corresponding label
    label = y_train[i]
    
    plt.title(f"Image {i+1}: Label: {label}")
    
    plt.show()


# In[12]:


unique_labels = np.unique(y_train)
print("Unique Labels in Training Dataset:", unique_labels)


# In[15]:


label_to_disease = {
    866: "Diabetic Retinopathy",
    867: "Cataract",
    868: "Glaucoma",
    # Add more mappings as needed
}

# ... (your existing code)

# Visualize a few example images with labels
num_images_to_visualize = 1500

for i in range(num_images_to_visualize):
    plt.imshow(X_train[i].reshape((64, 64, 3)))  # Assuming images are color (3 channels)
    
    # Set the title with the corresponding label
    label = y_train[i]
    
    if label in label_to_disease:
        disease = label_to_disease[label]
    else:
        disease = "Unknown Label"
    
    plt.title(f"Image {i+1}: Label: {label} ({disease})")
    
    plt.show()


# In[ ]:


unique_labels, label_counts = np.unique(y_train, return_counts=True)
label_to_count = dict(zip(unique_labels, label_counts))

print("Label Distribution in Training Dataset:")
for label, count in label_to_count.items():
    print(f"Label: {label}, Count: {count}")


# In[ ]:




