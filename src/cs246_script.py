from num_sys_class import *
from num_sys_class import _number_sys, _ieee754
import torch
import torch.nn.functional as F

def run_experiment(number_format, input, filter):
    formatted_input = number_format.real_to_format_tensor(input)
    formatted_filter = number_format.real_to_format_tensor(filter)
    return convolve(formatted_input, formatted_filter)

def convolve(input, filter):
    return F.conv2d(input, filter, padding=1)  # Assuming padding=1 for simplicity

if __name__ == '__main__':
    scale_factor = 100.
    print("scale factor => ", scale_factor)
    
    # Define dimensions for the random tensor
    batch_size = 16
    channels = 64  # Typical number of channels in intermediate layers of ResNet
    height = 32
    width = 32

    # Generate a random tensor with the specified dimensions
    input_tensor = torch.randn(batch_size, channels, height, width)
    input_tensor = input_tensor * scale_factor

    print(input_tensor.shape)
    # print(input_tensor)
    
    # Define the dimensions of the convolutional kernel
    in_channels = 64  # Number of input channels
    out_channels = 128  # Number of output channels
    kernel_size = 3  # Size of the kernel (3x3)

    # Generate a random convolutional filter/kernel
    conv_filter = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    print(conv_filter.shape)
    # print(conv_filter)

    # Perform convolution
    output_tensor = convolve(input_tensor, conv_filter)

    print(output_tensor.shape)  # Print the shape of the output tensor
    # print(output_tensor)
    

    mx_format_sweep = [
        [2,1,1],
        [2,1,4],
        [2,1,8],
        [2,1,16],
        [2,1,32],
        [2,1,64],
        [3,2,1],
        [3,2,4],
        [3,2,8],
        [3,2,32],
        [3,2,64],
        [4,3,1],
        [4,3,4],
        [4,3,8],
        [4,3,32],
        [4,3,64]
    ]
    print("format\te\tm\tk\tloss")    
    for params in mx_format_sweep:
        number_format = mx_float(
            exp_len=params[0],
            mant_len=params[1],
            n_blocks=params[2]
        )
        result = run_experiment(number_format, input_tensor, conv_filter)
        loss = torch.nn.functional.l1_loss(result, output_tensor)
        print("mxfloat\t", '\t'.join(map(str, params + [loss])))
        
    adaptiv_format_sweep = [
        [2,1,'-'],
        [3,2,'-'],
        [4,3,'-'],
    ]
    for params in adaptiv_format_sweep:
        number_format = adaptive_float(
            bit_width=params[0]+params[1]+1,
            exp_len=params[0],
            mant_len=params[1]
        )
        result = run_experiment(number_format, input_tensor, conv_filter)
        loss = torch.nn.functional.l1_loss(result, output_tensor)
        print("afloat\t", '\t'.join(map(str, params + [loss])))
        
            
    block_format_sweep = [
        [2,1,'-'],
        [3,2,'-'],
        [4,3,'-'],
    ]
    for params in block_format_sweep:
        number_format = block_fp(
            bit_width=params[0]+params[1]+1,
            exp_len=params[0],
            mant_len=params[1]
        )
        result = run_experiment(number_format, input_tensor, conv_filter)
        loss = torch.nn.functional.l1_loss(result, output_tensor)
        print("bfloat\t", '\t'.join(map(str, params + [loss])))
        
    fpn_format_sweep = [
        [2,1,'-'],
        [3,2,'-'],
        [4,3,'-'],
        [5,10,'-'],
        [8,23,'-'],
    ]
    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])
    for params in fpn_format_sweep:
        number_format = num_float_n(
            exp_len=params[0],
            mant_len=params[1]
        )
        result = run_experiment(number_format, input_tensor, conv_filter)
        loss = torch.nn.functional.l1_loss(result, output_tensor)
        print("fpn\t", '\t'.join(map(str, params + [loss])))
    