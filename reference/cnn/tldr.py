import pandas as pd
import torch
from tensorflow import keras
from tensorflow.keras import layers


"""
This script shows how to generally build and use a CNN model in Keras.
It should not be used as a reference for any specific details. For that, see the other scripts in this directory.
"""

input_shape = (28, 28, 1)  # TODO
x_train = None  # TODO
y_train = None  # TODO

model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=15,
)

history_df = pd.DataFrame(history.history)

"""
The same in pytorch:
"""

input_shape = (1, 28, 28)  # TODO
x_train = None  # TODO
y_train = None  # TODO


class MyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)  # in_channels, out_channels, kernel_size
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 64, 5)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_data_loader = torch.utils.data.DataLoader()  # TODO
test_data_loader = torch.utils.data.DataLoader()  # TODO

model = MyCNN()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
