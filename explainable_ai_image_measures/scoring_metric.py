import numpy as np
import torch
from sklearn.metrics import auc
import torch.nn.functional as F

from explainable_ai_image_measures.irof import IrofDataset
from explainable_ai_image_measures.pixel_relevancy import PixelRelevancyDataset


class Measures:
    def __init__(self,
                 model,
                 batch_size=64,
                 irof_segments=40,
                 irof_sigma=5,
                 pixel_package_size=1,
                 normalize=True,
                 clip01=False,
                 baseline_color=None):
        """
        Parametrize the future measurements

        model: PyTorch model
        batch_size: During each iteration batch_size number of images will be sent through the network simultaneously
        irof_segments: Maximum number of slic segments, that we want to use for measuring. Only relevant if you compute
                       IROF later
        irof_sigma: Parameter used in the slic algorithm
        pixel_package_size: E.g. for imagenet you may have 224*224=50,176 pixels. Therefore sending 50,176 pixels
                            through the network may lead to too much overhead. Instead you can also remove / add blocks
                            of pixels to speed up the computation. Only relevant for IAUC, DAUC
        normalize: With activated normalization the new probabilities are divided by the probabiilties of the old image.
                   This allows the comparison of attributions independent of how sure the network is for the original
                   image. Activating normalization is highly encouraged if comparing attributions across several
                   images.
        clip01: Clips the computed probabilities between [0, 1]. This is only relevant for normalize=True.
                In some cases the probabilities after e.g. removing parts of the original image may be higher than
                before. E.g. for IROF this could theoretically lead to negative scores. If you want to prohibit this,
                activate clip01. Note that the clipping clips each individual score. Indirectly you also ensure that
                the final score is within [0,1]
        baseline_color: For IROF and DAUC we iteratively remove parts of the image and replace it by the baseline
                        color as specified here. For IAUC we start with an image consisting only of the baseline_color.
                        By default the mean color is used.
        """

        self.model = model
        self.batch_size = batch_size
        self.irof_segments = irof_segments
        self.irof_sigma = irof_sigma
        self.pixel_package_size = pixel_package_size
        self.normalize = normalize
        self.clip01 = clip01
        self.baseline_color = baseline_color

    def _calc_probs(self, image_batch, label):
        probs = F.softmax(self.model(image_batch), dim=1)
        return probs[:, label]

    def _calc_single_score(self, scoring_dataset, label):
        probs = []
        with torch.no_grad():
            for j, img_batch in enumerate(scoring_dataset):
                probs += [self._calc_probs(img_batch, label)]
        probs = torch.cat(probs).flatten()

        if self.normalize:
            probs = probs[:-1] / probs[-1]
        else:
            probs = probs[:-1]

        if self.clip01:
            probs = torch.clamp(probs, 0, 1)

        probs = scoring_dataset.postprocess_scores(probs)

        x = np.arange(0, len(probs))
        y = probs.detach().cpu().numpy()
        score = auc(x, y) / len(probs)

        return score, probs.detach()

    def _assert_check(self, image, attribution):
        assert(len(image.shape) == 3)
        assert(image.shape[1:] == attribution.shape)
        if self.baseline_color is not None:
            assert(len(self.baseline_color.shape) == 1)
            assert(len(self.baseline_color) == image.shape[0])

    def compute_IAUC(self, image, attribution, label):
        """
        Computes IAUC for a single image and attribution
        image: Torch.FloatTensor(color_channel, width, height)
        attribution: Torch.FloatTensor(width, height)
        label: Label of the attribution
        """
        self._assert_check(image, attribution)

        with torch.no_grad():
            dataset = PixelRelevancyDataset(
                image,
                attribution,
                True,
                self.batch_size,
                self.pixel_package_size,
                image.device,
                self.baseline_color
            )

            return self._calc_single_score(dataset, label)

    def compute_DAUC(self, image, attribution, label):
        """
        Computes DAUC for a single image and attribution
        image: Torch.FloatTensor(color_channel, width, height)
        attribution: Torch.FloatTensor(width, height)
        label: Label of the attribution
        """
        self._assert_check(image, attribution)

        with torch.no_grad():
            dataset = PixelRelevancyDataset(
                image,
                attribution,
                False,
                self.batch_size,
                self.pixel_package_size,
                image.device,
                self.baseline_color
            )
            return self._calc_single_score(dataset, label)

    def compute_IROF(self, image, attribution, label):
        """
        Computes IROF for a single image and attribution
        image: Torch.FloatTensor(color_channel, width, height)
        attribution: Torch.FloatTensor(width, height)
        label: Label of the attribution
        """
        self._assert_check(image, attribution)

        with torch.no_grad():
            dataset = IrofDataset(
                image,
                attribution,
                self.batch_size,
                self.irof_segments,
                self.irof_sigma,
                image.device,
                self.baseline_color
            )

            return self._calc_single_score(dataset, label)

    def compute_batch(self, images, attributions, labels, IROF=True, IAUC=True, DAUC=True):
        """
        Computes the batch for many images and allows multiple attributions per image.
        image: Torch.FloatTensor(nr_images, color_channel, width, height)
        attribution: (nr_images, nr_attributions_per_image, width, height)
        labels: Tuple / Array / Tensor of Int
        IROF: Defines, whether IROF is computed
        IAUC: Defines, whether IAUC is computed
        DAUC: Defines, whether DAUC is computed
        """
        assert(len(images) == len(attributions))
        assert(len(images) == len(labels))

        functions = dict()
        if IROF:
            functions["IROF"] = self.compute_IROF
        if IAUC:
            functions["IAUC"] = self.compute_IAUC
        if DAUC:
            functions["DAUC"] = self.compute_DAUC
        if len(functions) == 0:
            return None

        result = dict()
        for method in functions:
            scores = torch.zeros(attributions.shape[0:2])
            probs = []

            for img_id in range(len(images)):
                probs.append([])
                for attr_id in range(len(attributions[img_id])):
                    score, prob = functions[method](
                        images[img_id],
                        attributions[img_id, attr_id],
                        labels[img_id]
                    )

                    scores[img_id, attr_id] = score
                    probs[-1].append(prob)

                probs[-1] = torch.stack(probs[-1])
            result[method] = (scores, probs)

        return result
