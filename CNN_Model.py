{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "19a57bf0-a1c7-434a-a10b-6d67a865defc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0d0bac2b-ce97-464b-8dc8-a1f9a364fad0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Load the Data\n",
    "# We load the files you created in the previous step\n",
    "X = np.load(\"seismic_data_X.npy\")\n",
    "y = np.load(\"seismic_labels_y.npy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4c4135ff-2216-4ce7-8ad3-8cf4131c800d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert to PyTorch tensors (Float32 is standard for AI)\n",
    "# We need to reshape X to [Batch_Size, Channels, Length] -> [2000, 1, 100]\n",
    "X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) \n",
    "y_tensor = torch.tensor(y, dtype=torch.long)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f47e659a-b461-454e-ad5f-cb40ef248fda",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d1631ea1-c9a7-407f-b38c-06b34fd9def9",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dataset = TensorDataset(X_train, y_train)\n",
    "test_dataset = TensorDataset(X_test, y_test)\n",
    "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
    "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b153e5d7-048c-44cb-9eca-f0ab6b8a72d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "class SeismicCNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(SeismicCNN, self).__init__()\n",
    "        # Layer 1: Convolution (Feature Extraction)\n",
    "        # Input: 1 channel (signal), Output: 16 filters\n",
    "        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.pool = nn.MaxPool1d(kernel_size=2) # Reduces size by half (100 -> 50)\n",
    "        \n",
    "        # Layer 2: Another Convolution\n",
    "        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)\n",
    "        # Pool again (50 -> 25)\n",
    "        \n",
    "        # Fully Connected Layer (Classification)\n",
    "        # Input features: 32 channels * 25 length = 800\n",
    "        self.fc1 = nn.Linear(32 * 25, 2) # Output: 2 classes (Safe vs Fault)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pool(self.relu(self.conv1(x))) # Block 1\n",
    "        x = self.pool(self.relu(self.conv2(x))) # Block 2\n",
    "        x = x.view(x.size(0), -1) # Flatten for linear layer\n",
    "        x = self.fc1(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4a5e78f9-00a2-4ad0-9642-1beda9501a05",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = SeismicCNN()\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "41f18029-9511-431d-bba0-4ad4655f3658",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting Training...\n",
      "Epoch 1/10, Loss: 0.6843\n",
      "Epoch 2/10, Loss: 0.5884\n",
      "Epoch 3/10, Loss: 0.4811\n",
      "Epoch 4/10, Loss: 0.4709\n",
      "Epoch 5/10, Loss: 0.4623\n",
      "Epoch 6/10, Loss: 0.4594\n",
      "Epoch 7/10, Loss: 0.4584\n",
      "Epoch 8/10, Loss: 0.4587\n",
      "Epoch 9/10, Loss: 0.4550\n",
      "Epoch 10/10, Loss: 0.4557\n"
     ]
    }
   ],
   "source": [
    "num_epochs = 10\n",
    "print(\"Starting Training...\")\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for inputs, labels in train_loader:\n",
    "        optimizer.zero_grad()           # Clear old gradients\n",
    "        outputs = model(inputs)         # Forward pass\n",
    "        loss = criterion(outputs, labels) # Calculate error\n",
    "        loss.backward()                 # Backward pass (learn)\n",
    "        optimizer.step()                # Update weights\n",
    "        running_loss += loss.item()\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "248fa0ad-f012-49f7-8608-2327aee777bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final Accuracy: 77.50%\n",
      "Model weights saved as 'seismic_cnn_weights.pth'\n"
     ]
    }
   ],
   "source": [
    "model.eval()\n",
    "correct = 0\n",
    "total = 0\n",
    "with torch.no_grad():\n",
    "    for inputs, labels in test_loader:\n",
    "        outputs = model(inputs)\n",
    "        _, predicted = torch.max(outputs.data, 1)\n",
    "        total += labels.size(0)\n",
    "        correct += (predicted == labels).sum().item()\n",
    "\n",
    "accuracy = 100 * correct / total\n",
    "print(f\"Final Accuracy: {accuracy:.2f}%\")\n",
    "\n",
    "# 6. Save the Model Weights (Item #2 Deliverable)\n",
    "torch.save(model.state_dict(), \"seismic_cnn_weights.pth\")\n",
    "print(\"Model weights saved as 'seismic_cnn_weights.pth'\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (TF)",
   "language": "python",
   "name": "tf"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
