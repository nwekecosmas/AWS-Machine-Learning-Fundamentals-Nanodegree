import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.in_ch = 3

        self.layer1 = self._make_layer(64, blocks=1, stride=1)
        self.layer2 = self._make_layer(128, blocks=1, stride=2)
        self.layer3 = self._make_layer(256, blocks=1, stride=2)
        self.layer4 = self._make_layer(512, blocks=1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, out_ch, blocks, stride):
        layers = [ResidualBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)  # 64
        x = self.layer2(x)  # 128
        x = self.layer3(x)  # 256
        x = self.layer4(x)  # 512
        x = self.pool(x)
        x = self.classifier(x)
        return x


# import torch
# import torch.nn as nn


# # define the CNN architecture
# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()

#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))
        
#         self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(16),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2),
#                                    nn.Dropout2d(p=dropout))
        
#         self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(32),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2),
#                                    nn.Dropout2d(p=dropout))
        
#         self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(64),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2),
#                                    nn.Dropout2d(p=dropout))
 
#         self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(128),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2),
#                                    nn.Dropout2d(p=dropout))
    
#         self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(256),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2),
#                                    nn.Dropout2d(p=dropout))
        
        
#         self.mlp = nn.Sequential(nn.Linear(256 * 7 * 7, 1024),
#                                   nn.BatchNorm1d(1024),
#                                   nn.ReLU(),
#                                   nn.Dropout(p=dropout),
                                  
#                                   nn.Linear(1024, 512),
#                                   nn.BatchNorm1d(512),
#                                   nn.ReLU(),
#                                   nn.Dropout(p=dropout),
            
#                                   nn.Linear(512, num_classes))
            
  
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = torch.flatten(x, 1)
#         x = self.mlp(x)
        
#         return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
