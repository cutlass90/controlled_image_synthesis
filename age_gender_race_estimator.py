import torch
from torch import nn
import torch.nn.functional as F

from modules import ResnetBlock


class AgeGenderRaceEstimator(nn.Module):

    def __init__(self, img_size, n_downsamples=5, filters=64, RESNET_BLOCKS=[1, 2, 4, 2, 1], emb_size=512):
        super().__init__()

        self.n_downsamples = n_downsamples
        self.filters = filters
        self.emb_size = emb_size
        self.img_size = img_size

        assert len(RESNET_BLOCKS) == n_downsamples

        features = []
        for i, rb in zip(range(n_downsamples), RESNET_BLOCKS):
            inp = 3 if i == 0 else 2 ** (i - 1) * filters
            out = 2 ** i * filters
            features += [nn.Conv2d(inp, out, 3, 2, 1), nn.PReLU(out)]
            for _ in range(rb):
                features += [ResnetBlock(out, 'zeros', nn.BatchNorm2d, activation=nn.PReLU(out))]
        self.features = nn.Sequential(*features)

        self.age = nn.Sequential(
            nn.Linear(int(out * (img_size / 2 ** n_downsamples) ** 2), self.emb_size),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.SELU(),
            nn.Linear(self.emb_size, 100)
        )

        self.gender = nn.Sequential(
            nn.Linear(int(out * (img_size / 2 ** n_downsamples) ** 2), self.emb_size),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.SELU(),
            nn.Linear(self.emb_size, 2)
        )

        self.race = nn.Sequential(
            nn.Linear(int(out * (img_size / 2 ** n_downsamples) ** 2), self.emb_size),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.SELU(),
            nn.Linear(self.emb_size, 5)
        )

    def forward(self, x):
        if x.size(2) != self.img_size:
            x = nn.AdaptiveAvgPool2d(self.img_size)(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        age_logits = self.age(x)
        gender_logits = self.gender(x)
        race_logits = self.race(x)
        return age_logits, gender_logits, race_logits

if __name__ == "__main__":
    estimator = AgeGenderRaceEstimator(128)
    age, gender, race = estimator(torch.ones([4, 3, 128, 128]))
    print(age, gender, race)

