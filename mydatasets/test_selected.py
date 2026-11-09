import traceback
from bike import Bike
from imdb_large import IMDBLarge
from mnist2mnistm import Mnist2MnistM
from tinyimagenet import TinyImageNet
# 注意：如果你有 House.py 和 HR.py，请在这里加上导入
# from House import House
# from HR import HR

DATA_ROOT = "/root/autodl-tmp/EVA/data"

def test_dataset(name, func, *args, **kwargs):
    print(f"\n{'='*40}")
    print(f"正在测试: {name}")
    try:
        result = func(*args, **kwargs)
        
        # 提取 train, val, test 长度
        train_ds = result[-3]
        val_ds = result[-2] if len(result) >= 9 else "N/A"
        test_ds = result[-1]
        
        print(f"✅ {name} 加载成功！")
        print(f"   Train 样本数: {len(train_ds)}")
        if val_ds != "N/A" and val_ds is not None:
            print(f"   Val   样本数: {len(val_ds)}")
        print(f"   Test  样本数: {len(test_ds)}")
    except Exception as e:
        print(f"❌ {name} 加载失败！")
        print(f"   错误信息: {e}")
        # 如果想知道具体哪行报错，把下面这行的注释解开
        # traceback.print_exc()

if __name__ == "__main__":
    print("开始测试已有数据集...")
    
    # 1. 测试 Bike (表格数据)
    test_dataset("Bike", Bike, DATA_ROOT)
    
    # 2. 测试 IMDB Large (可能比较大，如果内存不够会崩)
    test_dataset("IMDBLarge", IMDBLarge, DATA_ROOT)
    
    # 3. 测试 TinyImageNet
    test_dataset("TinyImageNet", TinyImageNet, DATA_ROOT)
    
    # 4. 测试 Mnist2MnistM 
    # (注意：它内部会先下载 MNIST 到 DATA_ROOT，再去读 parquet，可能需要点时间)
    test_dataset("Mnist2MnistM", Mnist2MnistM, DATA_ROOT)
    
    # 5. 测试 House 和 HR (如果报错提示找不到模块，请检查文件名大小写)
    # test_dataset("House", House, DATA_ROOT)
    # test_dataset("HR", HR, DATA_ROOT)
    
    print("\n测试完毕！")
