"""Accounting checks without importing Torch or rerunning any experiment."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline as p
from export_case_timings import export


class CaseTimingTests(unittest.TestCase):
    def fixture(self,root):
        config=dict(datasets=['fixture'],fractions=[.1,.3],cops_variants=['wy','woy'],
                    seed=42,scientific_results=False)
        matrix=[case for case in p.cases(config) if case['baseline']=='COPS']
        p.save(root/'effective_config.json',dict(config))
        p.save(root/'prepare/COPS/fixture/complete.json',
               dict(dataset='fixture',baseline='COPS',status='ok',elapsed_sec=10.))
        for i,case in enumerate(matrix):
            directory=p.case_dir(root,case)
            selected=round(200*case['fraction'])
            p.save(directory/'selection.json',dict(**case,status='ok',elapsed_sec=2.+i,
                n_train=200,n_selected=selected,requested_count=selected,actual_fraction=case['fraction']))
            p.save(directory/'result.json',dict(**case,status='ok',elapsed_sec=6.,gpu='fixture',accuracy=.5))
        return config,matrix

    def test_each_configuration_gets_full_shared_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,matrix=self.fixture(root)
            records=p.summarize(root,matrix,config)
            timing=json.loads((root/'timing.json').read_text())
            aggregate=json.loads((root/'matrix_timing.json').read_text())
            self.assertEqual([row['total_sec'] for row in timing['results']],[18.,19.,20.,21.])
            self.assertEqual([row['total_sec'] for row in records],[18.,19.,20.,21.])
            self.assertEqual(aggregate['preparation_sec'],10.)
            self.assertEqual(sum(row['prepare_sec'] for row in timing['results']),40.)

    def test_missing_elapsed_is_rejected_not_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,matrix=self.fixture(root)
            path=p.case_dir(root,matrix[0])/'result.json'
            stage=json.loads(path.read_text());stage.pop('elapsed_sec');p.save(path,stage)
            with self.assertRaisesRegex(ValueError,'elapsed_sec'):
                p.recorded_case_timings(root,matrix,config)

    def test_missing_result_has_no_invented_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,matrix=self.fixture(root)
            (p.case_dir(root,matrix[0])/'result.json').unlink()
            records=p.recorded_case_timings(root,matrix,config)
            self.assertEqual(records[0]['status'],'missing')
            self.assertIsNone(records[0]['total_sec'])

    def test_export_preserves_raw_measurements_and_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,matrix=self.fixture(root)
            # Export the four fixture cases, keeping all experiment launch paths forbidden.
            p.save(root/'timing.json',dict(invocation_wall_sec=99.25))
            legacy=(root/'timing.json').read_bytes()
            raw={path:hashlib.sha256(path.read_bytes()).hexdigest() for pattern in
                 ('prepare/*/*/complete.json','*/*/fraction_*/*/selection.json','*/*/fraction_*/*/result.json')
                 for path in root.glob(pattern)}
            with (patch.object(p,'cases',return_value=matrix),
                  patch.object(p,'launch',side_effect=AssertionError('Training started')),
                  patch.object(p,'native',side_effect=AssertionError('Native training started'))):
                report=export(root)
            self.assertEqual(report['reconstructed_results'],4)
            self.assertFalse(report['training_rerun'])
            self.assertEqual((root/'matrix_timing.json').read_bytes(),legacy)
            for path,digest in raw.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main(verbosity=2)
