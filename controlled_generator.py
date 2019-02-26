from torch import nn
from modifier import Modifier
from age_gender_race_estimator import AgeGenderRaceEstimator
from GAN import Generator


class ControlledGenerator(nn.Module):

    def __init__(self, z_size, image_size):
        super().__init__()
        self.modifier = Modifier(z_size)
        self.estimator = AgeGenderRaceEstimator(image_size, n_downsamples=4, filters=64, RESNET_BLOCKS=[1, 2, 2, 1], emb_size=256)
        self.generator = Generator(3, 128, z_size, image_size)

    def forward(self, z, target_age, target_gender):
        base_img = self.generator(z)
        z = self.modifier(z, target_age, target_gender)
        img = self.generator(z)
        age_logits, gender_logits, race_logits = self.estimator(img)
        return base_img, img, age_logits, gender_logits, race_logits, z



