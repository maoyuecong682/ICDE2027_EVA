"""Bounded launcher checks with tiny subprocesses; no Torch or training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline as p


class LauncherTests(unittest.TestCase):
    def test_detached_run_holds_lock_and_preserves_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copyfile(p.HERE/'run_all.sh',root/'run_all.sh')
            (root/'runtime_check.py').write_text('raise SystemExit(0)\n')
            (root/'pipeline.py').write_text(
                "import os,time\nfrom pathlib import Path\n"
                "root=Path(os.environ['MATRIX_OUTPUT'])\n"
                "(root/'entered').write_text(str(os.getpid()))\n"
                "while not (root/'release').exists():time.sleep(.02)\n")
            output=root/'out'
            env=dict(os.environ,MATRIX_PYTHON=sys.executable,MATRIX_OUTPUT=str(output))
            env.pop('MATRIX_LOCK_HELD',None)
            first=subprocess.run(['bash',str(root/'run_all.sh')],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(first.returncode,0,first.stderr)
            try:
                deadline=time.monotonic()+5
                while not (output/'entered').exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue((output/'entered').exists())
                pid_before=(output/'launcher.pid').read_text()
                second=subprocess.run(['bash',str(root/'run_all.sh')],env=env,capture_output=True,text=True,timeout=5)
                self.assertEqual(second.returncode,75,second.stderr)
                self.assertEqual((output/'launcher.pid').read_text(),pid_before)
            finally:
                (output/'release').touch()
            # A completed run releases the lock, allowing a new invocation.
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                probe=subprocess.run(['flock','-n',str(output/'.run.lock'),'true'])
                if probe.returncode==0:break
                time.sleep(.02)
            self.assertEqual(probe.returncode,0)

    def test_timeout_kills_native_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stub=root/'worker.py'
            stub.write_text(
                'import subprocess,sys,time\nfrom pathlib import Path\n'
                'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\n'
                f'Path({str(root / "child.pid")!r}).write_text(str(child.pid))\n'
                'time.sleep(60)\n')
            args=SimpleNamespace(config=root/'config.json',timeout=1)
            result=root/'result.json'
            with patch.object(p,'__file__',str(stub)),patch.object(p,'release_idle_input_pages'):
                self.assertFalse(p.launch(args,{},root,'prepare',result,dataset='fixture'))
            self.assertEqual(json.loads(result.read_text())['returncode'],'timeout')
            pid=int((root/'child.pid').read_text())
            stat=Path('/proc')/str(pid)/'stat'
            deadline=time.monotonic()+2
            while stat.exists() and stat.read_text().split()[2]!='Z' and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')

    def test_completed_stage_is_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=Path(tmp)/'result.json'
            p.save(result,{'status':'ok'})
            with patch.object(p.subprocess,'Popen',side_effect=AssertionError('Stage repeated')):
                self.assertTrue(p.launch(None,{},Path(tmp),'prepare',result))

    def test_formal_dual_preserves_scheduler_and_has_no_smoke_limit(self):
        config=json.loads((p.HERE/'config.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p.save(root/'preflight/cifar100.json',dict(classes=100,selection_classes=100,
                   is_tabular=False,cuda=True,fingerprint='fixture'))
            with patch.object(p,'execute_sequence') as sequence:
                p.prepare(config,root,'cifar100','DUAL',root/'complete.json',0)
            steps=sequence.call_args.args[1]
            arguments=steps[0]['args']
            self.assertEqual(arguments[arguments.index('--epochs')+1],'200')
            self.assertEqual(arguments[arguments.index('--dynamics-epochs')+1],'30')
            self.assertNotIn('--max-update-batches',arguments)
            self.assertTrue(sequence.call_args.args[3]['scientific_results'])
            self.assertEqual(sequence.call_args.args[3]['max_update_batches'],0)
            self.assertIn('--dual-only',steps[1]['args'])


if __name__=='__main__':unittest.main(verbosity=2)
