import os
import json
import tempfile
import unittest
import numpy as np

from lib.data.dataset_precomputed import PrecomputedWHAMFolderDataset


class TestDatasetPrecomputed(unittest.TestCase):
    def _make_video_folder(self, root, vid, fragments, joints_len=120):
        folder = os.path.join(root, vid)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'summary.json'), 'w') as f:
            json.dump({'status': 'success', 'num_fragments': len(fragments)}, f)
        for fid in fragments:
            joints = np.random.randn(joints_len, 31, 3).astype(np.float32)
            verts = np.random.randn(joints_len, 20, 3).astype(np.float32)
            frame_ids = np.arange(joints_len)
            np.savez(os.path.join(folder, f'wham_fragment_id{fid}.npz'), joints=joints, verts=verts, frame_ids=frame_ids, fps=30.0)
            lma = np.random.randn(joints_len, 55).astype(np.float32)
            np.save(os.path.join(folder, f'lma_features_id{fid}.npy'), lma)
            np.save(os.path.join(folder, f'lma_dict_id{fid}.npy'), np.array({'dummy': np.ones((joints_len,))}, dtype=object))

    def test_precomputed_dataset_load(self):
        with tempfile.TemporaryDirectory() as td:
            datasets_root = os.path.join(td, 'datasets')
            root = os.path.join(datasets_root, 'NPDI_features')
            os.makedirs(root, exist_ok=True)
            self._make_video_folder(root, 'vPorn000001', [0, 1], joints_len=80)
            self._make_video_folder(root, 'vPorn000002', [5], joints_len=100)
            self._make_video_folder(root, 'vPorn000003', [7, 8, 9], joints_len=120)

            registry = {
                'global_seed': 123,
                'datasets': {
                    'npdi_tier3': {
                        'features_root': 'NPDI_features',
                        'label_tier': 3,
                        'include_failed': False,
                        'split_seed': 123,
                        'split_ratios': {'train': 0.67, 'val': 0.33, 'test': 0.0}
                    }
                }
            }
            reg_path = os.path.join(td, 'registry.json')
            with open(reg_path, 'w') as f:
                json.dump(registry, f)

            ds_train = PrecomputedWHAMFolderDataset(
                registry_path=reg_path,
                dataset_keys='npdi_tier3',
                data_split='train',
                datasets_root=datasets_root,
                n_frames=64,
                num_joints=17,
                scale_range=[1, 1],
            )

            motion, label, video_info = ds_train[0]
            self.assertEqual(motion.shape, (2, 64, 17, 3))
            self.assertEqual(label, 3)
            self.assertIn('fragment_id', video_info)
            self.assertIn('video_path', video_info)

    def test_precomputed_dataset_load_with_direct_features_root(self):
        with tempfile.TemporaryDirectory() as td:
            datasets_root = os.path.join(td, 'datasets')
            root = os.path.join(datasets_root, 'NPDI_features')
            os.makedirs(root, exist_ok=True)
            self._make_video_folder(root, 'vPorn000001', [0], joints_len=80)
            self._make_video_folder(root, 'vPorn000002', [1], joints_len=100)

            registry = {
                'global_seed': 123,
                'datasets': {
                    'npdi_tier3': {
                        'features_root': 'NPDI_features',
                        'label_tier': 3,
                        'include_failed': False,
                        'split_seed': 123,
                        'split_ratios': {'train': 1.0, 'val': 0.0, 'test': 0.0}
                    }
                }
            }
            reg_path = os.path.join(td, 'registry.json')
            with open(reg_path, 'w') as f:
                json.dump(registry, f)

            ds_train = PrecomputedWHAMFolderDataset(
                registry_path=reg_path,
                dataset_keys='npdi_tier3',
                data_split='train',
                datasets_root=root,
                n_frames=64,
                num_joints=17,
                scale_range=[1, 1],
            )

            motion, label, video_info = ds_train[0]
            self.assertEqual(motion.shape, (2, 64, 17, 3))
            self.assertEqual(label, 3)
            self.assertIn('fragment_id', video_info)
            self.assertIn('video_path', video_info)


if __name__ == '__main__':
    unittest.main()
