import torch
import numpy as np
from skimage.segmentation import slic

from explainable_ai_image_measures.pixel_manipulation import PixelManipulationBase


class IrofDataset(PixelManipulationBase):
    def __init__(
        self, image, attribution, batch_size, irof_segments, irof_sigma, device, baseline_color
    ):
        PixelManipulationBase.__init__(
            self, image, attribution, False, batch_size, device, baseline_color
        )
        self._irof_segments = irof_segments
        self._irof_sigma = irof_sigma

        self.generate_pixel_batches()
        self.generate_initial_image()

        self.generate_temp_baseline()

    def generate_pixel_batches(self):
        # Apply Slic algorithm to get superpixel areas
        img_np = self._image.detach().cpu().numpy().transpose(1, 2, 0)
        segments = slic(img_np, self._irof_segments, self._irof_segments).reshape(-1)
        nr_segments = np.max(segments) + 1
        segments = torch.LongTensor(segments).to(device=self._device)

        # Attribution score of each segment = Mean attribution
        attr = self._attribution.reshape(-1)
        seg_mean = [
            torch.mean(attr[segments == seg]).item() for seg in range(nr_segments)
        ]
        seg_mean = torch.FloatTensor(seg_mean).to(device=self._device)

        # Sort segments descending by mean attribution
        seg_rank = torch.argsort(-seg_mean)

        # Create lists of "shape" [batch_size, segment_size]
        # containing the indices of the segments
        self._pixel_batches = [[]]
        for seg in seg_rank:
            indices = torch.nonzero(segments == seg).flatten()
            IrofDataset._add_to_hierarhical_list(
                self._pixel_batches, self._batch_size, indices
            )

        # Add placeholder for original image
        IrofDataset._add_to_hierarhical_list(
            self._pixel_batches, self._batch_size, torch.Tensor([0]).to(device=self._device)
        )

    @staticmethod
    def _add_to_hierarhical_list(list_element, target_size, item):
        if len(list_element[-1]) == target_size:
            list_element.append([])
        list_element[-1].append(item)

    def _gen_indices(self, index):
        batch_size = self._get_batch_size(index)

        # Get all pixels
        all_pixels = torch.cat(self._pixel_batches[index]).to(self._device).long()

        # Create a matrix of indices of size [batch_size, all_pixels]
        template_indices = all_pixels.view(1, -1).repeat(batch_size, 1)

        # For each package only keep the previous pixels and package size additional pixels
        pixel_per_image = torch.LongTensor(
            [len(package) for package in self._pixel_batches[index]]
        ).to(self._device)
        cumsum = torch.cumsum(pixel_per_image, dim=0)
        keep_index_template = torch.cat(
            [torch.arange(0, s.item()) for s in cumsum]
        ).to(self._device)

        template_indices = template_indices.reshape(-1)[keep_index_template]
        template_indices = self._color_channel_shift(template_indices)

        batch_indices = self._batch_shift(template_indices, pixel_per_image)

        return template_indices, batch_indices

    def _get_fake_image_size(self):
        return 1

    def postprocess_scores(self, y):
        return 1-y
