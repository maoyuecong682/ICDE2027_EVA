# -*- coding: utf-8 -*-
"""verify_split_determinism.py —— 逐个数据集跑两次独立子进程，比对切分是否一致

为什么不用 shell 循环：上一版用 `python ... 2>/dev/null` 把错误也一并丢了，
结果第一遍里 cifar100/covtype 静默没有输出（很可能是被 cgroup OOM 杀掉），
diff 只显示"少了几行"，看不出是崩溃还是真的不一致。

这里改用 subprocess：每个数据集两次全新进程（不能在同一进程里跑两次，否则
eva_adapter 的 _CACHE 会把第二次直接返回缓存，永远"一致"）。子进程的
stdout/stderr 全部捕获，成功与否都能看见。

退出码：全一致 0，有不一致或任一次失败 1。
"""
import subprocess
import sys

DATASETS = ["cifar100", "covtype", "fashionmnist", "HR", "mnist",
            "qmnist", "SVHN", "tinyimagenet", "mnist2mnistm"]
PY = "/root/miniconda3/bin/python"
SCRIPT = "/root/eva_output/check_split_determinism.py"

ok_all = True
for ds in DATASETS:
    runs = []
    for k in (1, 2):
        p = subprocess.run([PY, "-u", SCRIPT, ds], capture_output=True, text=True)
        if p.returncode != 0:
            print("[%s] 第 %d 次失败 rc=%d" % (ds, k, p.returncode))
            tail = (p.stderr or p.stdout).strip().splitlines()[-4:]
            for line in tail:
                print("    | " + line)
            ok_all = False
            runs = None
            break
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("[")]
        runs.append([ln.split("] ", 1)[1] for ln in lines if ln.startswith("[%s]" % ds)])

    if runs is None:
        continue
    # 只比 labels=<hash>：顺序敏感的标签序列能同时反映成员与顺序，是切分是否
    # 一致的充分信号。feats 不用来判定 —— cifar100 / SVHN / tinyimagenet 的
    # train transform 带随机增广，同一份切分取两次特征本来就会不同。
    def _sig(rows):
        return [r.split(" labels=")[1].split(" ")[0] if " labels=" in r else r for r in rows]

    if _sig(runs[0]) == _sig(runs[1]):
        print("[%s] 一致  %s" % (ds, " / ".join(runs[0])))
    else:
        ok_all = False
        print("[%s] **不一致**" % ds)
        for a, b in zip(runs[0], runs[1]):
            print("    1) %s" % a)
            print("    2) %s" % b)

print("\nRESULT %s" % ("ALL-STABLE" if ok_all else "UNSTABLE"))
sys.exit(0 if ok_all else 1)
