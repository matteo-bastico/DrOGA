import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNBinaryClassification(nn.Module):
    def __init__(self, input_features = 35, ch1=32, ch2=128, fc1=2048, fc2=512, fc3=128):
        super(CNNBinaryClassification, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(1, ch1, 1),
            nn.BatchNorm1d(ch1),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(ch1, ch2, 1),
            nn.BatchNorm1d(ch2),
            nn.ReLU()
        )

        self.fc_layer1 = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(input_features * ch2, fc1),
            nn.BatchNorm1d(fc1),
            nn.ReLU()
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(fc1, fc2),
            nn.BatchNorm1d(fc2),
            nn.ReLU()
        )

        self.fc_layer3 = nn.Sequential(
            nn.Linear(fc2, fc3),
            nn.BatchNorm1d(fc3),
            nn.ReLU()
        )

        self.out_layer = nn.Linear(fc3, 1)

        self.double()

    def forward(self, inputs):

        x = self.conv_layer1(inputs)
        x = self.conv_layer2(x)
        x = x.view((x.size(0), -1))
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return self.out_layer(x)


class CNNBinaryClassificationSkip(nn.Module):
    def __init__(self, input_features=35, ch1=32, ch2=128, fc1=2048, fc2=512, fc3=128, sk2=16, sk1=128):
        super(CNNBinaryClassificationSkip, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(1, ch1, 1),
            nn.BatchNorm1d(ch1),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(ch1, ch2, 1),
            nn.BatchNorm1d(ch2),
            nn.ReLU()
        )

        self.fc_layer1 = nn.Sequential(
            nn.Linear(input_features * ch2, fc1),
            nn.BatchNorm1d(fc1),
            nn.ReLU()
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(input_features*sk2 + fc1, fc2),
            nn.BatchNorm1d(fc2),
            nn.ReLU()
        )

        self.fc_layer3 = nn.Sequential(
            nn.Linear(fc2 + sk1, fc3),
            nn.BatchNorm1d(fc3),
            nn.ReLU()
        )

        self.skip_conn2 = nn.Sequential(
            nn.Conv1d(ch1, sk2, 1),
            nn.BatchNorm1d(sk2),
            nn.ReLU()
        )

        self.skip_conn1 = nn.Sequential(
            nn.Linear(input_features, sk1),
            nn.BatchNorm1d(sk1),
            nn.ReLU()
        )

        self.out_layer = nn.Linear(fc3, 1)

        self.double()

    def forward(self, inputs):
        # Change shape of vector input
        inputs = inputs.view(inputs.shape[0], 1, -1)

        x = self.conv_layer1(inputs)
        skip1 = self.skip_conn1(inputs.view(inputs.shape[0], -1))
        skip2 = self.skip_conn2(x)
        skip2 = skip2.view((skip2.size(0), -1))
        x = self.conv_layer2(x)
        x = x.view((x.size(0), -1))
        x = self.fc_layer1(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.fc_layer2(x)
        
        x = torch.cat((x, skip1), dim=1)

        x = self.fc_layer3(x)
        return self.out_layer(x)