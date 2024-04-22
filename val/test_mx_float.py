from goldeneye.src.num_sys_class import *
from goldeneye.src.num_sys_class import _number_sys, _ieee754
import torch
import math

# MX Flot
def test_mx_float():
    # test tensors to use on different systems
    # 1. small numbers
    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])

    # 2. large numbers
    test2 = torch.tensor([[ 997.481,  188.034, -147.376, -277.766],
                          [-617.844, -755.696,   18.283,  670.539],
                          [-709.682, -841.260,  300.587,  837.047],
                          [ 347.082,   98.871, -775.379,  709.284]])
    
    # 3. mix of small and large numbers
    test3 = torch.tensor([
        [[-1.17,  2.71, -1.60,  0.43],
         [ 997.481,  188.034, -147.376, -277.766]],
        [[-1.14,  2.05,  1.01,  0.07],
         [-617.844, -755.696,   18.283,  670.539]],
        [[ 0.16, -0.03, -0.89, -0.87],
         [-709.682, -841.260,  300.587,  837.047]],
        [[-0.04, -0.39,  0.64, -2.89],
         [347.082,   98.871, -775.379,  709.284]]
    ])
       
    # FP4 (E2M1)
    mxfp4_e2m1_k4 = mx_float(
        exp_len=2,
        mant_len=1,
        n_blocks=4
    )
    expected = torch.tensor([
            [-1.0000,  2.0000, -1.5000,  0.0000],
            [-1.0000,  2.0000,  1.0000,  0.0000],
            [ 0.1250, -0.0000, -0.7500, -0.7500],
            [-0.0000, -0.0000,  0.5000, -2.0000]
        ]) 
    assert(torch.equal(
        mxfp4_e2m1_k4.real_to_format_tensor(test1),
        expected
    ))
    expected = torch.tensor([
            [ 768.,  128., -128., -256.],
            [-512., -512.,    0.,  512.],
            [-512., -768.,  256.,  768.],
            [ 256.,    0., -768.,  512.]
        ]) 
    assert(torch.equal(
        mxfp4_e2m1_k4.real_to_format_tensor(test2),
        expected
    ))
    expected = torch.tensor([
        [[-1.0000e+00,  2.0000e+00, -1.5000e+00,  0.0000e+00],
         [ 7.6800e+02,  1.2800e+02, -1.2800e+02, -2.5600e+02]],
        [[-1.0000e+00,  2.0000e+00,  1.0000e+00,  0.0000e+00],
         [-5.1200e+02, -5.1200e+02,  0.0000e+00,  5.1200e+02]],
        [[ 1.2500e-01, -0.0000e+00, -7.5000e-01, -7.5000e-01],
         [-5.1200e+02, -7.6800e+02,  2.5600e+02,  7.6800e+02]],
        [[-0.0000e+00, -0.0000e+00,  5.0000e-01, -2.0000e+00],
         [ 2.5600e+02,  0.0000e+00, -7.6800e+02,  5.1200e+02]]
    ])
    assert(torch.equal(
        mxfp4_e2m1_k4.real_to_format_tensor(test3),
        expected
    ))
    
    # =====================
    
    # FP6 (E3M2)
    mxfp6_e3m2_k4 = mx_float(
        exp_len=3,
        mant_len=2,
        n_blocks=4
    )
    expected = torch.tensor([
            [-1.0000,  2.5000, -1.5000,  0.3750],
            [-1.0000,  2.0000,  1.0000,  0.0625],
            [ 0.1562, -0.0273, -0.8750, -0.7500],
            [-0.0391, -0.3750,  0.6250, -2.5000]
        ]) 
    assert(torch.allclose(
        mxfp6_e3m2_k4.real_to_format_tensor(test1),
        expected,
        rtol=0.01,
        atol=0.0001
    ))  
    expected = torch.tensor([
            [ 896.,  160., -128., -256.],
            [-512., -640.,   16.,  640.],
            [-640., -768.,  256.,  768.],
            [ 320.,   96., -768.,  640.]
        ]) 
    assert(torch.allclose(
        mxfp6_e3m2_k4.real_to_format_tensor(test2),
        expected,
        rtol=0.01,
        atol=0.0001
    ))  
    expected = torch.tensor([
        [[-1.0000e+00,  2.5000e+00, -1.5000e+00,  3.7500e-01],
         [ 8.9600e+02,  1.6000e+02, -1.2800e+02, -2.5600e+02]],
        [[-1.0000e+00,  2.0000e+00,  1.0000e+00,  6.2500e-02],
         [-5.1200e+02, -6.4000e+02,  1.6000e+01,  6.4000e+02]],
        [[ 1.5625e-01, -2.7344e-02, -8.7500e-01, -7.5000e-01],
         [-6.4000e+02, -7.6800e+02,  2.5600e+02,  7.6800e+02]],
        [[-3.9062e-02, -3.7500e-01,  6.2500e-01, -2.5000e+00],
         [ 3.2000e+02,  9.6000e+01, -7.6800e+02,  6.4000e+02]]
    ])
    assert(torch.allclose(
        mxfp6_e3m2_k4.real_to_format_tensor(test3),
        expected,
        rtol=0.01,
        atol=0.0001
    ))  
    mxfp6_e3m2_k8 = mx_float(
        exp_len=3,
        mant_len=2,
        n_blocks=8
    )
    expected = torch.tensor([
        [[  -0.,    0.,   -0.,    0.],
         [ 896.,  160., -128., -256.]],
        [[  -0.,    0.,    0.,    0.],
         [-512., -640.,   16.,  640.]],
        [[   0.,   -0.,   -0.,   -0.],
         [-640., -768.,  256.,  768.]],
        [[  -0.,   -0.,    0.,   -0.],
         [ 320.,   96., -768.,  640.]]
    ])
    assert(torch.allclose(
        mxfp6_e3m2_k8.real_to_format_tensor(test3),
        expected,
        rtol=0.01,
        atol=0.0001
    ))
    
    # =====================
    
    # FP8 (E4M2)
    mxfp8_e4m3_k4 = mx_float(
        exp_len=4,
        mant_len=3,
        n_blocks=4
    )
    expected = torch.tensor([
            [-1.1250,  2.5000, -1.5000,  0.4062],
            [-1.1250,  2.0000,  1.0000,  0.0625],
            [ 0.1562, -0.0293, -0.8750, -0.8125],
            [-0.0391, -0.3750,  0.6250, -2.7500]
        ]) 
    assert(torch.allclose(
        mxfp8_e4m3_k4.real_to_format_tensor(test1),
        expected,
        rtol=0.01,
        atol=0.0001
    ))
    expected = torch.tensor([
            [ 960.,  176., -144., -256.],
            [-576., -704.,   18.,  640.],
            [-704., -832.,  288.,  832.],
            [ 320.,   96., -768.,  704.]
        ]) 
    assert(torch.allclose(
        mxfp8_e4m3_k4.real_to_format_tensor(test2),
        expected,
        rtol=0.01,
        atol=0.0001
    ))
    expected = torch.tensor([
        [[-1.1250e+00,  2.5000e+00, -1.5000e+00,  4.0625e-01],
         [ 9.6000e+02,  1.7600e+02, -1.4400e+02, -2.5600e+02]],
        [[-1.1250e+00,  2.0000e+00,  1.0000e+00,  6.2500e-02],
         [-5.7600e+02, -7.0400e+02,  1.8000e+01,  6.4000e+02]],
        [[ 1.5625e-01, -2.9297e-02, -8.7500e-01, -8.1250e-01],
         [-7.0400e+02, -8.3200e+02,  2.8800e+02,  8.3200e+02]],
        [[-3.9062e-02, -3.7500e-01,  6.2500e-01, -2.7500e+00],
         [ 3.2000e+02,  9.6000e+01, -7.6800e+02,  7.0400e+02]]
    ])
    assert(torch.allclose(
        mxfp8_e4m3_k4.real_to_format_tensor(test3),
        expected,
        rtol=0.01,
        atol=0.0001
    ))  
    mxfp8_e4m3_k8 = mx_float(
        exp_len=4,
        mant_len=3,
        n_blocks=8
    )
    expected = torch.tensor([
        [[-1.1250e+00,  2.5000e+00, -1.5000e+00,  4.0625e-01],
         [ 9.6000e+02,  1.7600e+02, -1.4400e+02, -2.5600e+02]],
        [[-1.1250e+00,  2.0000e+00,  1.0000e+00,  6.2500e-02],
         [-5.7600e+02, -7.0400e+02,  1.8000e+01,  6.4000e+02]],
        [[ 1.5625e-01, -0.0000e+00, -8.7500e-01, -8.1250e-01],
         [-7.0400e+02, -8.3200e+02,  2.8800e+02,  8.3200e+02]],
        [[-3.9062e-02, -3.7500e-01,  6.2500e-01, -2.7500e+00],
         [ 3.2000e+02,  9.6000e+01, -7.6800e+02,  7.0400e+02]]
    ])
    assert(torch.allclose(
        mxfp8_e4m3_k8.real_to_format_tensor(test3),
        expected,
        rtol=0.01,
        atol=0.0001
    ))  
    