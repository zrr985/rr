#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN�����汾���ű�
���ڼ��RKNN-Toolkit2���������̼��汾һ����
"""

import sys
import os
import subprocess
import platform

def check_python_version():
    """���Python�汾"""
    print("=" * 50)
    print("Python������Ϣ")
    print("=" * 50)
    print(f"Python�汾: {sys.version}")
    print(f"Python·��: {sys.executable}")
    print(f"ƽ̨�ܹ�: {platform.machine()}")
    print(f"����ϵͳ: {platform.system()} {platform.release()}")
    print()

def check_rknn_toolkit():
    """���RKNN-Toolkit2�汾"""
    print("=" * 50)
    print("RKNN-Toolkit2�汾���")
    print("=" * 50)
    
    try:
        import rknnlite
        print(f"rknnlite�汾: {rknnlite.__version__}")
        print(f"rknnlite·��: {rknnlite.__file__}")
    except ImportError as e:
        print(f"? �޷�����rknnlite: {e}")
        return False
    except Exception as e:
        print(f"? ���rknnliteʱ����: {e}")
        return False
    
    try:
        from rknnlite.api import RKNNLite
        print("? RKNNLite API ����")
    except ImportError as e:
        print(f"? �޷�����RKNNLite API: {e}")
        return False
    
    return True

def check_system_libraries():
    """���ϵͳ��汾"""
    print("=" * 50)
    print("ϵͳ����")
    print("=" * 50)
    
    # ���librknnrt.so
    try:
        result = subprocess.run(['find', '/usr/lib', '/lib', '/opt', '-name', 'librknnrt.so*', '2>/dev/null'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0 and result.stdout.strip():
            print("? �ҵ�librknnrt.so:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
                    # ���Ի�ȡ���ļ���Ϣ
                    try:
                        info_result = subprocess.run(['file', line], capture_output=True, text=True)
                        if info_result.returncode == 0:
                            print(f"    ��Ϣ: {info_result.stdout.strip()}")
                    except:
                        pass
        else:
            print("? δ�ҵ�librknnrt.so")
    except Exception as e:
        print(f"? ���librknnrt.soʱ����: {e}")
    
    # ���NPU�豸
    try:
        if os.path.exists('/dev/rknpu'):
            print("? �ҵ�NPU�豸: /dev/rknpu")
        else:
            print("? δ�ҵ�NPU�豸: /dev/rknpu")
    except Exception as e:
        print(f"? ���NPU�豸ʱ����: {e}")

def check_model_info():
    """���ģ����Ϣ"""
    print("=" * 50)
    print("ģ����Ϣ���")
    print("=" * 50)
    
    model_path = "./rknnModel/md1.rknn"
    if not os.path.exists(model_path):
        print(f"? ģ���ļ�������: {model_path}")
        return False
    
    print(f"? ģ���ļ�����: {model_path}")
    print(f"�ļ���С: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # ���Լ���ģ�ͻ�ȡ��ϸ��Ϣ
    try:
        from rknnlite.api import RKNNLite
        rknn_lite = RKNNLite()
        
        print("���ڼ���ģ��...")
        ret = rknn_lite.load_rknn(model_path)
        if ret != 0:
            print(f"? ����ģ��ʧ��: {ret}")
            return False
        
        print("? ģ�ͼ��سɹ�")
        
        # ��ȡģ����Ϣ
        try:
            inputs_info = rknn_lite.query_input_info()
            print(f"���������: {len(inputs_info)}")
            for i, input_info in enumerate(inputs_info):
                print(f"  ����� {i}: ��״={input_info.shape}, ����={input_info.dtype}")
        except Exception as e:
            print(f"?? �޷���ȡ������Ϣ: {e}")
        
        try:
            outputs_info = rknn_lite.query_output_info()
            print(f"���������: {len(outputs_info)}")
            for i, output_info in enumerate(outputs_info):
                print(f"  ����� {i}: ��״={output_info.shape}, ����={output_info.dtype}")
        except Exception as e:
            print(f"?? �޷���ȡ�����Ϣ: {e}")
        
        # ���Գ�ʼ������ʱ
        print("���ڳ�ʼ������ʱ...")
        ret = rknn_lite.init_runtime()
        if ret != 0:
            print(f"? ��ʼ������ʱʧ��: {ret}")
            rknn_lite.release()
            return False
        
        print("? ����ʱ��ʼ���ɹ�")
        rknn_lite.release()
        
    except Exception as e:
        print(f"? ���ģ��ʱ����: {e}")
        return False
    
    return True

def check_environment_variables():
    """��黷������"""
    print("=" * 50)
    print("�����������")
    print("=" * 50)
    
    env_vars = ['LD_LIBRARY_PATH', 'MVCAM_SDK_PATH', 'MVCAM_COMMON_RUNENV']
    
    for var in env_vars:
        value = os.environ.get(var, '')
        if value:
            print(f"? {var}: {value}")
        else:
            print(f"? {var}: δ����")

def main():
    """������"""
    print("RKNN������Ϲ���")
    print("=" * 50)
    
    # ���Python����
    check_python_version()
    
    # ���RKNN-Toolkit2
    if not check_rknn_toolkit():
        print("? RKNN-Toolkit2���ʧ�ܣ����鰲װ")
        return
    
    # ���ϵͳ��
    check_system_libraries()
    
    # ��黷������
    check_environment_variables()
    
    # ���ģ��
    if check_model_info():
        print("\n? ģ�ͼ��ͨ��")
    else:
        print("\n? ģ�ͼ��ʧ��")
    
    print("\n" + "=" * 50)
    print("������")
    print("=" * 50)

if __name__ == "__main__":
    main()
