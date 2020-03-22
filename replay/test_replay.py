import abc
import unittest
import numpy as np
import torch
from replay import ram_buffer, vram_buffer
from replay.sampler import weighted_sampler, offset_sampler


class TestBufferBasics(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_buffer(self):
        pass

    def test_ring_buffer(self):
        buffer = self.build_buffer()
        buffer.add_samples(np.array([[1, 1], [2, 2]]))
        np.testing.assert_almost_equal(buffer.to_array(), np.array([[1, 1], [2, 2]]), decimal=2)

        buffer.add_samples(np.array([[3, 3], [4, 4]]))
        buffer.add_samples(np.array([[5, 5], [6, 6]]))

        np.testing.assert_almost_equal(buffer.to_array(), np.array([[6, 6], [2, 2], [3, 3], [4, 4], [5, 5]]),
                                       decimal=2)

        buffer.add_samples(np.array([[7, 7], [8, 8]]))
        buffer.add_samples(np.array([[9, 9], [10, 10]]))

        np.testing.assert_almost_equal(buffer.to_array(), np.array([[6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]),
                                       decimal=2)
        buffer.add_samples(np.array([[11, 11]]))
        np.testing.assert_almost_equal(buffer.to_array(), np.array([[11, 11], [7, 7], [8, 8], [9, 9], [10, 10]]),
                                       decimal=2)

    def test_sampling(self):
        buffer = self.build_buffer()
        buffer.add_samples(np.array([[1., 1], [2., 2]]))
        buffer.add_samples(np.array([[3., 3], [4., 4]]))
        buffer.add_samples(np.array([[5., 5], [6., 6]]))

        samples = buffer.sample(100000).cpu().numpy()
        np.testing.assert_equal(samples.shape, (100000, 2))
        np.testing.assert_equal(samples[:, 0], samples[:, 1])
        np.testing.assert_(np.all(samples[:, 0] != 1))
        num_2s = sum(samples[:, 0] == 2)
        for i in range(2, 7):
            np.testing.assert_(sum(np.isclose(samples[:, 0], i)) > 0, f"{i}")
        np.testing.assert_(25000 > num_2s > 15000)

        sequence_samples = buffer.sample_sequence(100, 3).cpu().numpy()
        np.testing.assert_equal(sequence_samples.shape, (100, 3, 2))
        np.testing.assert_almost_equal((sequence_samples[:, 0, 0] - 2) % 5, (sequence_samples[:, 2, 0] - 4) % 5,
                                       decimal=2)


class TestCpuRamBuffer(unittest.TestCase, TestBufferBasics):
    def build_buffer(self):
        self._device = torch.device('cpu')
        return ram_buffer.RamBuffer(5, 2, device=self._device)

    def test_indexing(self):
        buffer = self.build_buffer()
        buffer.add_samples(np.array([[1, 1], [2, 2]]))
        buffer.add_samples(np.array([[3, 3], [4, 4]]))
        buffer.add_samples(np.array([[5, 5], [6, 6]]))
        buffer.add_samples(np.array([[7, 7], [8, 8]]))
        buffer.add_samples(np.array([[9, 9], [10, 10]]))
        buffer.add_samples(np.array([[11, 11]]))

        self.assertAlmostEqual(buffer[0][0], 7)
        np.testing.assert_almost_equal(buffer[0, :], np.array([7, 7]))
        np.testing.assert_almost_equal(buffer[0, 1:], np.array([7]))
        before, after = buffer[1:3, 1:]
        np.testing.assert_almost_equal(before, np.array([[8], [9]]))
        np.testing.assert_(after is None)
        before, after = buffer[0:, 0]
        np.testing.assert_almost_equal(before, np.array([7, 8, 9, 10]))
        np.testing.assert_almost_equal(after, np.array([11]))
        before, after = buffer[0:5, 0]
        np.testing.assert_almost_equal(before, np.array([7, 8, 9, 10]))
        np.testing.assert_almost_equal(after, np.array([11]))
        before, after = buffer[0:4, 0]
        np.testing.assert_almost_equal(before, np.array([7, 8, 9, 10]))
        np.testing.assert_(after is None)
        before, after = buffer[3:3, 0]
        np.testing.assert_equal(before.shape[0], 0)
        np.testing.assert_(after is None)
        before, after = buffer[:, :]
        np.testing.assert_almost_equal(before, np.array([[7, 7], [8, 8], [9, 9], [10, 10]]))
        np.testing.assert_almost_equal(after, np.array([[11, 11]]))
        np.testing.assert_almost_equal(buffer[np.array([0, 2, 4]), 0], np.array([7, 9, 11]))

        buffer[:3, :] = np.array([[10, 10], [11, 11], [12, 12]])
        np.testing.assert_almost_equal(buffer.to_array()[:, 0], np.array([11, 10, 11, 12, 10]))

        buffer[2:, :] = np.array([[22, 22], [23, 23], [24, 14]])
        np.testing.assert_almost_equal(buffer.to_array()[:, 0], np.array([24, 10, 11, 22, 23]))
        buffer[0, :] = np.array([[30, 30]])
        np.testing.assert_almost_equal(buffer.to_array()[:, 0], np.array([24, 30, 11, 22, 23]))


class TestGpuRamBuffer(unittest.TestCase, TestBufferBasics):
    def build_buffer(self):
        self._device = torch.device('cuda:0')
        return ram_buffer.RamBuffer(5, 2, device=self._device)


class TestVramBuffer(unittest.TestCase, TestBufferBasics):
    def build_buffer(self):
        self._device = torch.device('cuda:0')
        return vram_buffer.VramBuffer(5, 2, device=self._device)


class TestWeightedSampler(unittest.TestCase):
    def test_sample_weights(self):
        replay = ram_buffer.RamBuffer(5, 4, torch.device('cpu'))
        sampler = weighted_sampler.WeightedSampler(replay, 2, 3)
        replay.add_samples(np.array([
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [2, 2, 2, 0]
        ]))
        replay.add_samples(np.array([
            [3, 3, 0, 0],
            [4, 4, 3, 0],
        ]))

        samples = sampler.sample_weighted(100000).detach().cpu().numpy()
        sample_bincount = np.bincount(samples[:, 0].astype(np.int64))
        self.assertTrue(2.2 > sample_bincount[2]/sample_bincount[0] > 1.8)
        self.assertTrue(3.3 > sample_bincount[4]/sample_bincount[1] > 2.7)
        self.assertEqual(sample_bincount[3], 0)
        replay.add_samples(np.array([
            [5, 0, 1, 0],
            [6, 1, 3, 0],
        ]))
        samples = sampler.sample_weighted(100000).detach().cpu().numpy()
        sample_bincount = np.bincount(samples[:, 1].astype(np.int64))
        self.assertTrue(2.2 > sample_bincount[2]/sample_bincount[0] > 1.8)
        self.assertTrue(1.2 > sample_bincount[4]/sample_bincount[1] > 0.8)
        self.assertEqual(sample_bincount[3], 0)


class TestOffsetSampler(unittest.TestCase):
    def test_uniform_hard_cutoff(self):
        replay = ram_buffer.RamBuffer(14, 4, torch.device('cpu'))
        sampler = offset_sampler.OffsetSampler(replay, 5, 1, 2, 3, 0.5, False)

        sampler.add_episode(np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [6, 0, 0, 0],
        ]))
        sampler.add_episode(np.array([
            [10, 0, 0, 0],
            [11, 0, 0, 0],
            [12, 0, 0, 0],
            [13, 0, 0, 0],
            [14, 0, 0, 0],
            [15, 0, 0, 0],
            [16, 0, 0, 0],
            [17, 0, 0, 0],
        ]))

        samples, offsets = sampler.sample_uniform_offset(10000)
        valid_nums = [1, 2, 10, 11, 12, 13]
        counts = np.array([(samples[:, 0, 0] == i).sum() for i in valid_nums])
        self.assertTrue((samples[:, 0, 0] // 10 == samples[:, 1, 0] // 10).all().item())
        self.assertTrue(np.all(counts > 0))
        self.assertTrue(counts.sum() == 10000)
        self.assertTrue(np.all(2300 > np.bincount(offsets)))
        self.assertTrue(np.all(np.bincount(offsets) > 1700))

    def test_uniform_soft_cutoff(self):
        replay = ram_buffer.RamBuffer(14, 4, torch.device('cpu'))
        sampler = offset_sampler.OffsetSampler(replay, 5, 1, 2, 3, 0.5, True)

        sampler.add_episode(np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [6, 0, 0, 0],
        ], dtype=np.float32))
        sampler.add_episode(np.array([
            [10, 0, 0, 0],
            [11, 0, 0, 0],
            [12, 0, 0, 0],
            [13, 0, 0, 0],
            [14, 0, 0, 0],
            [15, 0, 0, 0],
            [16, 0, 0, 0],
            [17, 0, 0, 0],
        ], dtype=np.float32))

        samples, offsets = sampler.sample_uniform_offset(10000)
        valid_nums = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17]
        counts = np.array([(samples[:, 0, 0] == i).sum() for i in valid_nums])
        self.assertTrue((samples[:, 0, 0] // 10 == samples[:, 1, 0] // 10).all().item())
        self.assertTrue(np.all(counts > 0))
        self.assertTrue(counts.sum() == 10000)
        offset_bincount = np.bincount(offsets)
        for i in range(4):
            self.assertTrue(offset_bincount[i + 1] < offset_bincount[i])

    def test_geometric_hard_cutoff(self):
        replay = ram_buffer.RamBuffer(14, 4, torch.device('cpu'))
        sampler = offset_sampler.OffsetSampler(replay, 5, 1, 2, 3, 0.5, False)

        sampler.add_episode(np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [6, 0, 0, 0],
        ]))
        sampler.add_episode(np.array([
            [10, 0, 0, 0],
            [11, 0, 0, 0],
            [12, 0, 0, 0],
            [13, 0, 0, 0],
            [14, 0, 0, 0],
            [15, 0, 0, 0],
            [16, 0, 0, 0],
            [17, 0, 0, 0],
        ]))

        samples, offsets = sampler.sample_discounted_offset(10000)
        valid_nums = [1, 2, 10, 11, 12, 13]
        counts = np.array([(samples[:, 0, 0] == i).sum() for i in valid_nums])
        self.assertTrue((samples[:, 0, 0] // 10 == samples[:, 1, 0] // 10).all().item())
        self.assertTrue(np.all(counts > 0))
        self.assertTrue(counts.sum() == 10000)
        offset_bincount = np.bincount(offsets)
        for i in range(4):
            self.assertTrue(2.2 > offset_bincount[i]/offset_bincount[i + 1] > 1.8)

    def test_geometric_soft_cutoff(self):
        replay = ram_buffer.RamBuffer(14, 4, torch.device('cpu'))
        sampler = offset_sampler.OffsetSampler(replay, 5, 1, 2, 3, 0.5, True)

        sampler.add_episode(np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [6, 0, 0, 0],
        ]))
        sampler.add_episode(np.array([
            [10, 0, 0, 0],
            [11, 0, 0, 0],
            [12, 0, 0, 0],
            [13, 0, 0, 0],
            [14, 0, 0, 0],
            [15, 0, 0, 0],
            [16, 0, 0, 0],
            [17, 0, 0, 0],
        ]))

        samples, offsets = sampler.sample_discounted_offset(10000)
        valid_nums = [1, 2, 10, 11, 12, 13]
        counts = np.array([(samples[:, 0, 0] == i).sum() for i in valid_nums])
        self.assertTrue((samples[:, 0, 0] // 10 == samples[:, 1, 0] // 10).all().item())
        self.assertTrue(np.all(counts > 0))
        self.assertTrue(counts.sum() == 10000)
        offset_bincount = np.bincount(offsets)
        for i in range(4):
            self.assertTrue(2.2 > offset_bincount[i]/offset_bincount[i + 1])


if __name__ == '__main__':
    unittest.main()
