#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吸烟检测配置验证脚本
检查所有相关文件和配置是否正确
"""

import os
import sys

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} (文件不存在)")
        return False

def check_import_module(module_name, description):
    """检查模块是否可以导入"""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} (导入失败: {e})")
        return False

def verify_smoke_detection_config():
    """验证吸烟检测配置"""
    print("=" * 50)
    print("吸烟检测配置验证")
    print("=" * 50)
    
    # 检查必需文件
    required_files = [
        ("./smoking.rknn", "吸烟检测模型文件"),
        ("./func_smoke.py", "吸烟检测功能文件"),
        ("./rknnpool_smoke_single.py", "吸烟检测RKNN池文件"),
        ("./maincopy.py", "主程序文件"),
    ]
    
    files_ok = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            files_ok = False
    
    print("\n" + "-" * 50)
    
    # 检查必需模块
    required_modules = [
        ("cv2", "OpenCV库"),
        ("numpy", "NumPy库"),
        ("rknnlite.api", "RKNN Lite库"),
    ]
    
    modules_ok = True
    for module_name, description in required_modules:
        if not check_import_module(module_name, description):
            modules_ok = False
    
    print("\n" + "-" * 50)
    
    # 检查功能模块
    try:
        from func_smoke import CLASSES, myFunc_smoke
        print(f"✓ 吸烟检测类别定义: {CLASSES}")
        print(f"✓ 吸烟检测函数: myFunc_smoke")
        
        # 验证类别数量
        if len(CLASSES) == 3 and 'cigarette' in CLASSES and 'face' in CLASSES and 'smoking' in CLASSES:
            print("✓ 类别定义正确 (cigarette, face, smoking)")
        else:
            print("✗ 类别定义不正确，应该是 (cigarette, face, smoking)")
            modules_ok = False
            
    except ImportError as e:
        print(f"✗ 吸烟检测模块导入失败: {e}")
        modules_ok = False
    
    print("\n" + "-" * 50)
    
    # 检查RKNN池模块
    try:
        from rknnpool_smoke_single import rknnPoolExecutor_smoke
        print("✓ 吸烟检测RKNN池模块导入成功")
    except ImportError as e:
        print(f"✗ 吸烟检测RKNN池模块导入失败: {e}")
        modules_ok = False
    
    print("\n" + "=" * 50)
    
    if files_ok and modules_ok:
        print("🎉 所有配置检查通过！吸烟检测功能已准备就绪。")
        print("\n使用说明:")
        print("1. 运行主程序: python maincopy.py")
        print("2. 启动吸烟检测任务: 通过UDP命令启动任务代号 5")
        print("3. 测试功能: python test_smoke_detection.py")
        return True
    else:
        print("❌ 配置检查失败，请解决上述问题后重试。")
        return False

if __name__ == "__main__":
    success = verify_smoke_detection_config()
    sys.exit(0 if success else 1) 