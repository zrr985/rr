#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN环境版本检查脚本
用于检查RKNN-Toolkit2、驱动、固件版本一致性
"""

import sys
import os
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("Python环境信息")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"平台架构: {platform.machine()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print()

def check_rknn_toolkit():
    """检查RKNN-Toolkit2版本"""
    print("=" * 50)
    print("RKNN-Toolkit2版本检查")
    print("=" * 50)
    
    try:
        import rknnlite
        print(f"rknnlite版本: {rknnlite.__version__}")
        print(f"rknnlite路径: {rknnlite.__file__}")
    except ImportError as e:
        print(f"? 无法导入rknnlite: {e}")
        return False
    except Exception as e:
        print(f"? 检查rknnlite时出错: {e}")
        return False
    
    try:
        from rknnlite.api import RKNNLite
        print("? RKNNLite API 可用")
    except ImportError as e:
        print(f"? 无法导入RKNNLite API: {e}")
        return False
    
    return True

def check_system_libraries():
    """检查系统库版本"""
    print("=" * 50)
    print("系统库检查")
    print("=" * 50)
    
    # 检查librknnrt.so
    try:
        result = subprocess.run(['find', '/usr/lib', '/lib', '/opt', '-name', 'librknnrt.so*', '2>/dev/null'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0 and result.stdout.strip():
            print("? 找到librknnrt.so:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
                    # 尝试获取库文件信息
                    try:
                        info_result = subprocess.run(['file', line], capture_output=True, text=True)
                        if info_result.returncode == 0:
                            print(f"    信息: {info_result.stdout.strip()}")
                    except:
                        pass
        else:
            print("? 未找到librknnrt.so")
    except Exception as e:
        print(f"? 检查librknnrt.so时出错: {e}")
    
    # 检查NPU设备
    try:
        if os.path.exists('/dev/rknpu'):
            print("? 找到NPU设备: /dev/rknpu")
        else:
            print("? 未找到NPU设备: /dev/rknpu")
    except Exception as e:
        print(f"? 检查NPU设备时出错: {e}")

def check_model_info():
    """检查模型信息"""
    print("=" * 50)
    print("模型信息检查")
    print("=" * 50)
    
    model_path = "./rknnModel/md1.rknn"
    if not os.path.exists(model_path):
        print(f"? 模型文件不存在: {model_path}")
        return False
    
    print(f"? 模型文件存在: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # 尝试加载模型获取详细信息
    try:
        from rknnlite.api import RKNNLite
        rknn_lite = RKNNLite()
        
        print("正在加载模型...")
        ret = rknn_lite.load_rknn(model_path)
        if ret != 0:
            print(f"? 加载模型失败: {ret}")
            return False
        
        print("? 模型加载成功")
        
        # 获取模型信息
        try:
            inputs_info = rknn_lite.query_input_info()
            print(f"输入层数量: {len(inputs_info)}")
            for i, input_info in enumerate(inputs_info):
                print(f"  输入层 {i}: 形状={input_info.shape}, 类型={input_info.dtype}")
        except Exception as e:
            print(f"?? 无法获取输入信息: {e}")
        
        try:
            outputs_info = rknn_lite.query_output_info()
            print(f"输出层数量: {len(outputs_info)}")
            for i, output_info in enumerate(outputs_info):
                print(f"  输出层 {i}: 形状={output_info.shape}, 类型={output_info.dtype}")
        except Exception as e:
            print(f"?? 无法获取输出信息: {e}")
        
        # 尝试初始化运行时
        print("正在初始化运行时...")
        ret = rknn_lite.init_runtime()
        if ret != 0:
            print(f"? 初始化运行时失败: {ret}")
            rknn_lite.release()
            return False
        
        print("? 运行时初始化成功")
        rknn_lite.release()
        
    except Exception as e:
        print(f"? 检查模型时出错: {e}")
        return False
    
    return True

def check_environment_variables():
    """检查环境变量"""
    print("=" * 50)
    print("环境变量检查")
    print("=" * 50)
    
    env_vars = ['LD_LIBRARY_PATH', 'MVCAM_SDK_PATH', 'MVCAM_COMMON_RUNENV']
    
    for var in env_vars:
        value = os.environ.get(var, '')
        if value:
            print(f"? {var}: {value}")
        else:
            print(f"? {var}: 未设置")

def main():
    """主函数"""
    print("RKNN环境诊断工具")
    print("=" * 50)
    
    # 检查Python环境
    check_python_version()
    
    # 检查RKNN-Toolkit2
    if not check_rknn_toolkit():
        print("? RKNN-Toolkit2检查失败，请检查安装")
        return
    
    # 检查系统库
    check_system_libraries()
    
    # 检查环境变量
    check_environment_variables()
    
    # 检查模型
    if check_model_info():
        print("\n? 模型检查通过")
    else:
        print("\n? 模型检查失败")
    
    print("\n" + "=" * 50)
    print("诊断完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
