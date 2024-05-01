import os
import random
import torch
from torch.utils.cpp_extension import load
# from qtorch.quant import float_quantize, fixed_point_quantize, block_quantize

# import C++ code
current_path = os.path.dirname(os.path.realpath(__file__))
num_sys = load(
    name="num_sys",
    sources=[
        os.path.join(current_path, "num_sys.cpp"),
        os.path.join(current_path, "num_sys_helper.cpp"),
    ]
)

def float_quantize(float_value, man=23, exp=8):
    # Define scale and range for mantissa and exponent
    mantissa_range = 2 ** man
    exponent_range = 2 ** exp

    # Extract mantissa and exponent
    exponent = torch.floor(torch.log2(torch.abs(float_value)))
    mantissa = float_value / (2.0 ** exponent)

    # Ensure leading 1 in mantissa
    mantissa = torch.sign(mantissa) * torch.abs(mantissa)  # Ensure positive mantissa
    mantissa *= 0.5  # Scale mantissa to have leading 1

    # Quantize mantissa
    quantized_mantissa = torch.round(mantissa * mantissa_range) / mantissa_range
    quantized_mantissa = quantized_mantissa * 2. # account for leading 1

    # Compute exponent adjustment
    max_exponent = 2 ** (exp - 1) - 1
    exponent_adjustment = torch.clamp(exponent, -max_exponent, max_exponent)
    
    # Quantize exponent
    quantized_exponent = torch.round(exponent_adjustment * exponent_range) / exponent_range
    
    # print(quantized_exponent, quantized_mantissa)

    return quantized_mantissa * 2.0 ** quantized_exponent

def float_quantize2(tensor, man=23, exp=8):
    sign = torch.sign(tensor)
    float_arr = torch.abs(tensor)
    
    mant, exp = torch.frexp(float_arr)
    print(mant)
    
    # new mantissa
    mant = 2 * mant
    scale = 2 ** (-man)
    mant = ((mant / scale).round()) * scale
    
    print(scale)
    print(mant)
    
    return sign * mant * 2. ** (exp - 1)
    
    
def float_quantize1(tensor, man=4, exp=4):
    # Define scale and range for mantissa and exponent
    mantissa_range = 2 ** man # 2^1=2
    exponent_range = 2 ** exp # 2^2=4

    # Extract mantissa and exponent
    sign = torch.sign(tensor)
    exponent = torch.floor(torch.log2(torch.abs(tensor)))
    mantissa = torch.abs(tensor) / (2.0 ** exponent)
    print(mantissa * 2**exponent)

    # Ensure leading zero in mantissa
    mantissa *= 2  # Scale mantissa to have leading zero

    # Quantize mantissa
    quantized_mantissa = torch.round(mantissa * (mantissa_range - 1))

    # Compute exponent adjustment
    max_exponent = 2 ** (exp - 1) - 1
    exponent_adjustment = torch.clamp(exponent, -max_exponent, max_exponent)
    
    # Quantize exponent
    quantized_exponent = torch.round(exponent_adjustment * (exponent_range - 1))

    print(quantized_exponent, quantized_mantissa)

    return sign * quantized_mantissa * 2 ** quantized_exponent

class _number_sys:
    """
    General class for number systems, used to bit_flip using a specific format
    """
    def bit_flip(self, bit_arr, bit_ind):
        # interpret index from least significant bit
        bit_ind_LSB = len(bit_arr) - 1 - bit_ind

        # bit_arr to bit_arr
        bit_arr[bit_ind_LSB] = "0" if int(bit_arr[bit_ind_LSB]) else "1"
        return bit_arr

    def real_to_format(self, num):
        raise NotImplementedError

    def real_to_format_tensor(self, tensor):
        raise NotImplementedError

    def real_to_format_tensor_meta(self, tensor):
        raise NotImplementedError

    def format_to_real(self, bit_arr):
        raise NotImplementedError

    def format_to_real_tensor(self, tensor):
        return tensor.to(torch.float32)

    def single_bit_flip_in_format(self, num, bit_ind):
        bit_arr = self.real_to_format(num)
        assert 0 <= bit_ind < len(bit_arr), "bit index out of range"
        bit_arr_corrupted = self.bit_flip(bit_arr, bit_ind)

        return self.format_to_real(bit_arr_corrupted)

    def convert_numsys_flip(self, num, bit_ind, flip=False):
        bit_arr = self.real_to_format(num)

        if flip:
            bit_arr = self.bit_flip(bit_arr, bit_ind)

        return self.format_to_real(bit_arr)

    def convert_numsys_tensor(self, tensor, meta_inj=False):
        if meta_inj:
            return self.format_to_real_tensor(self.real_to_format_tensor_meta(tensor))
        else:
            return self.format_to_real_tensor(self.real_to_format_tensor(tensor))

    # HELPER FUNCTIONS
    def int_to_bin(num):
        # integer to its binary representation
        return str(bin(num))[2:]

    def frac_to_bin(frac):
        # a fraction (form: 0.sth) into its binary representation
        # exp: 0.5 -> "1", 0.25 -> "01", 0.125 -> "001"

        # declaring an empty string to store binary bits
        binary = str()

        # iterating through fraction until it becomes zero
        while frac:
            # multiplying fraction by 2
            frac *= 2

            # storing integer part of fraction in int_part
            if frac >= 1:
                int_part = 1
                frac -= 1
            else:
                int_part = 0

            # adding int_part to binary after every iteration
            binary += str(int_part)

        # returning the binary string
        return binary

    def bin_to_frac(frac_str):
        # a binary form to a fraction: "01" -> 0.25
        power_count = -1
        frac = 0

        for i in frac_str:
            frac += int(i) * pow(2, power_count)
            power_count -= 1

        # returning mantissa in 0.M form
        return frac


class _ieee754(_number_sys):
    """IEEE Standard 754 Floating Point Number System"""
    def __init__(
            self,
            exp_len=8,
            mant_len=23,
            bias=None,
            denorm=True,
            max_val=None,
            min_val=None
    ):
        self.exp_len = exp_len
        self.mant_len = mant_len
        self.bias = bias
        self.denorm = denorm
        self.max_val = max_val
        self.min_val = min_val

    def real_to_format(self, num):
        # compute bias
        if self.bias is None:
            self.bias = 2 ** (self.exp_len - 1) - 1

        # handle denorm
        if not self.denorm and self.max_val is not None and self.min_val is not None:
            if num < self.min_val:
                num = 0
            elif num > self.max_val:
                num = self.max_val

        # real to bit_arr
        sign = "1" if num < 0 else "0"

        num = abs(num)

        int_str = _number_sys.int_to_bin(int(num))
        frac_str = _number_sys.frac_to_bin(num - int(num))

        # init values
        exp_str = "0" * self.exp_len

        if int_str.find("1") != -1:
            # decimal shift
            ind = len(int_str) - 1 - int_str.index("1")
            int_str = int_str[len(int_str) - ind - 1:]
            exp_str = _number_sys.int_to_bin(ind + self.bias)
        else:
            if frac_str.find("1") != -1:
                dec_shift = frac_str.index("1") + 1
                if dec_shift > self.bias:
                    frac_str = frac_str[self.bias:]
                else:
                    exp_str = _number_sys.int_to_bin(-dec_shift + self.bias)
                    frac_str = frac_str[dec_shift:]

        mant_str = int_str[1:] + frac_str
        # zero padding
        exp_str = ("0" * (self.exp_len - len(exp_str))) + exp_str
        mant_str = (mant_str + ("0" * (self.mant_len - len(mant_str))))[: self.mant_len]

        # asserts
        assert len(exp_str) == self.exp_len,\
            "exp_len unknown error: %d != %d" %(len(exp_str), self.exp_len)
        assert len(mant_str) == self.mant_len,\
            "mant_len unknown error: %d != %d" %(len(mant_str), self.mant_str)

        return list("".join([sign, exp_str, mant_str]))

    def format_to_real(self, bit_arr):
        def mant_to_int(mantissa_str):
            # mantissa in 1.M form
            return _number_sys.bin_to_frac(mantissa_str) + 1

        # compute bias
        if self.bias is None:
            self.bias = 2 ** (self.exp_len - 1) - 1

        sign = pow(-1, int(bit_arr[0]))

        exp_str = "".join(bit_arr[1:self.exp_len + 1])
        exp = int(exp_str, 2) - self.bias

        mant_str = "".join(bit_arr[self.exp_len + 1:])
        mant = mant_to_int(mant_str)

        # exceptions
        if exp_str == "0" * self.exp_len and mant_str == "0" * self.mant_len:
            return 0
        if exp_str == "1" * self.exp_len and mant_str == "0" * self.mant_len:
            return sign * float('inf')
        if exp_str == "1" * self.exp_len and mant_str != "0" * self.mant_len:
            return float('nan')

        # handling denormals
        if exp_str == "0" * self.exp_len and mant_str != "0" * self.mant_len:
            if self.denorm:
                # denormalized
                mant -= 1
            else:
                # not using denormals (like in AdaptivFloat)
                mant = 0

        return sign * mant * pow(2, exp)

    def int_to_bitstream(self, num):
        # sign-magnitude is used for representing the sign
        sign = "1" if num < 0 else "0"
        num = abs(num)
        int_str = _number_sys.int_to_bin(int(num))
        if len(int_str) > self.exp_len:
            int_str = "1" * self.exp_len

        # zero padding
        int_str = ("0" * (self.exp_len - len(int_str))) + int_str
        return list(int_str)

    def bitstream_to_int(self, bit_arr):
        exp_str = "".join(bit_arr[1: self.exp_len + 1])
        exp = int(exp_str, 2)
        return exp


class num_fp32(_ieee754):
    """Floating Point 32 Number System"""
    def __init__(self):
        super(num_fp32, self).__init__()

    def real_to_format_tensor(self, tensor):
        return tensor.to(torch.float32)


class num_fp16(_ieee754):
    """Floating Point 16 Number System"""
    def __init__(self):
        super(num_fp16, self).__init__(exp_len=5, mant_len=10)

    def real_to_format_tensor(self, tensor):
        return tensor.to(torch.float16)


class num_float_n(_ieee754):
    """Floating Point Number System"""
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(self, exp_len=5, mant_len=10):
        super(num_float_n, self).__init__(exp_len=exp_len, mant_len=mant_len)

    def real_to_format_tensor(self, tensor):
        return float_quantize(tensor, exp=self.exp_len, man=self.mant_len)


class num_bfloat16(_ieee754):
    """Brain Float Number System"""
    def __init__(self):
        super(num_bfloat16, self).__init__(exp_len=8, mant_len=7)

    def real_to_format_tensor(self, tensor):
        return tensor.to(torch.bfloat16)


class num_fixed_pt(_number_sys):
    """Fixed Point Number System"""
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(self, int_len=3, frac_len=3):
        self.int_len = int_len
        self.frac_len = frac_len

    def real_to_format(self, num):
        # sign-magnitude is used for representing the sign
        sign = "1" if num < 0 else "0"
        num = abs(num)
        int_str = _number_sys.int_to_bin(int(num))
        if len(int_str) > self.int_len:
            int_str = "1" * self.int_len

        frac_str = _number_sys.frac_to_bin(num - int(num))[:self.frac_len]

        # zero padding
        int_str = ("0" * (self.int_len - len(int_str))) + int_str
        frac_str = frac_str + ("0" * (self.frac_len - len(frac_str)))

        return list(sign) + list(int_str) + list(frac_str)

    def real_to_format_tensor(self, tensor):
        return fixed_point_quantize(tensor,
                                    1 + self.int_len + self.frac_len,
                                    self.frac_len)

    def format_to_real(self, bit_arr):
        int_str, frac_str = map(lambda arr: "".join(arr),
                                (bit_arr[1:self.int_len + 1],
                                 bit_arr[self.int_len + 1:]),)
        sign = 1 if bit_arr[0] == "0" else -1
        return sign * (int(int_str, 2) + _number_sys.bin_to_frac(frac_str))


class block_fp(_ieee754):
    """Block Float Number System"""
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(self, bit_width=32, exp_len=8, mant_len=23):
        super(block_fp, self).__init__(exp_len=exp_len, mant_len=mant_len)
        self.bit_width = bit_width

    def real_to_format_tensor(self, tensor):
        return self.quant_bfloat(float_arr=tensor,
                                 n_bits=self.bit_width,
                                 n_exp=self.exp_len)

    def real_to_format_tensor_meta(self, tensor):
        return self.quant_bfloat_meta(float_arr=tensor,
                                      n_bits=self.bit_width,
                                      n_exp=self.exp_len)

    def quant_bfloat_py(self, float_arr, n_bits, n_exp):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2
        max_exp = 2 ** (n_exp - 1) - 1

        min_value = 2 ** min_exp
        max_value = (2 ** max_exp) * (2 - 2 ** (-n_mant))

        # non-denormal part
        float_arr[float_arr < min_value] = 0

        # 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        shared_exp = exp.max()
        exp_diff = shared_exp - exp
        power_exp_diff = torch.exp2(exp_diff)
        mant_adj = mant / power_exp_diff

        exp_adj = torch.full(exp.shape, shared_exp, device=float_arr.device)

        # exp should not be larger than max_exp
        assert (shared_exp <= max_exp)
        power_exp = torch.exp2(exp_adj)

        # 4. quantize mantissa
        scale = 2 ** (-n_mant)  # e.g. 2 bit, scale = 0.25
        mant_adj = ((mant_adj / scale).round()) * scale

        bfloat_out = sign * power_exp * mant_adj
        return bfloat_out

    def quant_bfloat(self, float_arr, n_bits=8, n_exp=3):
        # C++
        return num_sys.quant_bfloat(float_arr, n_bits, n_exp)

        # Python
        # return self.quant_bfloat_py(float_arr, n_bits, n_exp)

    def quant_bfloat_meta_py(self, float_arr, n_bits=8, n_exp=3):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2
        max_exp = 2 ** (n_exp - 1) - 1

        min_value = 2 ** min_exp
        max_value = (2 ** max_exp) * (2 - 2 ** (-n_mant))

        # non-denormal part
        float_arr[float_arr < min_value] = 0

        # 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        shared_exp = exp.max()

        # ============= ERROR INJECTION INTO META =============
        # get bit array of shared exp
        exp_str = self.int_to_bitstream(shared_exp)

        # flip a random bit
        bit_ind = random.randint(0, self.exp_len - 1)
        bit_arr = self.bit_flip(exp_str, bit_ind)

        # get numerical value
        shared_exp = self.bitstream_to_int(bit_arr)
        # ============= ERROR INJECTION INTO META =============

        exp_diff = shared_exp - exp
        power_exp_diff = torch.exp2(exp_diff)
        mant_adj = mant / power_exp_diff

        exp_adj = torch.full(exp.shape, shared_exp, device=float_arr.device)

        # exp should not be larger than max_exp
        assert (shared_exp <= max_exp)
        power_exp = torch.exp2(exp_adj)

        # 4. quantize mantissa
        scale = 2 ** (-n_mant)  # e.g. 2 bit, scale = 0.25
        mant_adj = ((mant_adj / scale).round()) * scale

        bfloat_out = sign * power_exp * mant_adj
        return bfloat_out

    def quant_bfloat_meta(self, float_arr, n_bits=8, n_exp=3):
        # C++
        return num_sys.quant_bfloat_meta(float_arr, n_bits, n_exp)

        # Python
        # return self.quant_bfloat_meta_py(float_arr, n_bits, n_exp)


class adaptive_float(_ieee754):
    """Adaptive Float Number System"""
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(self, bit_width=32, exp_len=8, mant_len=23, exp_bias=None):
        super(adaptive_float, self).__init__(exp_len=exp_len, mant_len=mant_len)
        self.bit_width = bit_width
        self.exp_bias = exp_bias

    def real_to_format_tensor(self, tensor):
        return self.quantize_adaptivfloat(float_arr=tensor,
                                          n_bits=self.bit_width,
                                          n_exp=self.exp_len,
                                          bias=self.exp_bias)

    def real_to_format_tensor_meta(self, tensor):
        return self.quantize_adaptivfloat_meta(float_arr=tensor,
                                               n_bits=self.bit_width,
                                               n_exp=self.exp_len,
                                               bias=self.exp_bias)

    def quantize_adaptivfloat_py(self, float_arr, n_bits=8, n_exp=4, bias=None):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        bias_temp = torch.frexp(float_arr.max())[1] - 1
        bias = (2 ** (n_exp - 1) - 1) - bias_temp

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2 - bias
        max_exp = 2 ** (n_exp - 1) - 1 - bias

        min_value = 2. ** min_exp
        max_value = (2. ** max_exp) * (2 - 2 ** (-n_mant))
        # non-denormal part
        float_arr[float_arr < min_value] = 0

        # 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        power_exp = torch.exp2(exp)

        # 4. quantize mantissa
        scale = 2 ** (-n_mant)  # e.g. 2 bit, scale = 0.25
        mant = ((mant / scale).round()) * scale

        float_out = sign * power_exp * mant
        return float_out

    def quantize_adaptivfloat(self, float_arr, n_bits=8, n_exp=4, bias=None):
        # C++
        if bias is None:
            return num_sys.quantize_adaptivfloat(float_arr, n_bits, n_exp, -1)
        else:
            return num_sys.quantize_adaptivfloat(float_arr, n_bits, n_exp, bias)

        # Python
        # return self.quantize_adaptivfloat_py(float_arr, n_bits, n_exp, bias)

    def quantize_adaptivfloat_meta_py(self,
                                      float_arr,
                                      n_bits=8,
                                      n_exp=4,
                                      bias=None):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        bias_temp = torch.frexp(float_arr.max())[1] - 1
        bias_in = (2 ** (n_exp - 1) - 1) - bias_temp

        # ============= ERROR INJECTION INTO META =============
        # get bit array of shared exp
        exp_str = self.int_to_bitstream(bias_in)

        # flip a random bit
        bit_ind = random.randint(0, 7)
        bit_arr = self.bit_flip(exp_str, bit_ind)

        # get numerical value
        bias = self.bitstream_to_int(bit_arr)
        # ============= ERROR INJECTION INTO META =============

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2 - bias
        max_exp = 2 ** (n_exp - 1) - 1 - bias

        min_value = 2. ** min_exp
        max_value = (2. ** max_exp) * (2 - 2 ** (-n_mant))
        # non-denormal part
        float_arr[float_arr < min_value] = 0

        # 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        power_exp = torch.exp2(exp)

        # 4. quantize mantissa
        scale = 2 ** (-n_mant)  # e.g. 2 bit, scale = 0.25
        mant = ((mant / scale).round()) * scale

        float_out = sign * power_exp * mant
        return float_out

    def quantize_adaptivfloat_meta(self,
                                   float_arr,
                                   n_bits=8,
                                   n_exp=4,
                                   bias=None):
        # C++
        return num_sys.quantize_adaptivfloat_meta(float_arr, n_bits, n_exp, -1)

        # Python
        # return self.quantize_adaptivfloat_meta_py(float_arr, n_bits, n_exp, bias)

class mx_float(_ieee754):
    """MX FLoat Number System"""
    def __init__(self, bit_width=32, exp_len=8, mant_len=23, n_blocks=4):
        super().__init__(exp_len=exp_len, mant_len=mant_len)
        self.bit_width = bit_width
        self.n_blocks = n_blocks
    
    def real_to_format_tensor(self, tensor):
        return self.quantize_mx_float(tensor)
        
    def real_to_format_tensor_meta(self, tensor):
        return self.quantize_mx_float(tensor, flip_meta=True)
        
    def quantize_mx_float(self, tensor, flip_meta=False):
        tensor_shape = tensor.shape

        # Flatten tensor so we can iterate through each element        
        tensor_flattened = tensor.flatten()
        
        # Get sign
        sign = torch.sign(tensor_flattened)
        float_arr = torch.abs(tensor_flattened)
        
        # Extract mantissa and exponent
        mant0, exp0 = torch.frexp(float_arr)
        
         # Calculate new mantissa
        mant_increment = 2 ** (-self.mant_len)
        mant0 = 2 * mant0 # IEEE mantissa implicit leading 1
        mant = ((mant0 / mant_increment).floor()) * mant_increment
        
        # Get constants
        exp_bias = 2. ** (self.exp_len - 1) - 1
        max_mant = 2. - 2. ** (-self.mant_len)
        max_exp = 2. ** self.exp_len - 2 - exp_bias
        max_value = (2. ** max_exp) * max_mant
        
        # Initialize new scale and exponent tensors
        exp = torch.zeros(float_arr.shape)
        scale = torch.zeros(float_arr.shape)
        
        # iterate through scaling blocks to calculate scale and exponent
        for i in range(0, len(float_arr), self.n_blocks): 
            # get block of float to process
            float_block = float_arr[i:i+self.n_blocks]
            
            # (max exp in block) - (offset for range of element) - (IEEE mant leading 1)
            shared_exp = torch.frexp(float_block.max())[1] - max_exp - 1
            scale_factor = 2. ** shared_exp
                                
            # Flip a bit in the shared_exp
            # ============= ERROR INJECTION INTO META =============
            if flip_meta:
                # TODO: also have to select which scaling block to flip
                pass
            # ============= ERROR INJECTION INTO META =============
            
            # Iterate through each value and populate
            for j in range(i, i+self.n_blocks, 1):
                quantized_value = float_arr[j] / scale_factor
                exp[j] = exp0[j] - shared_exp - 1
                if exp[j] < -exp_bias: # set subnormal values to zero
                    exp[j] = 0
                    mant[j] = 0
                elif quantized_value > max_value: # clamp larger values
                    exp[j] = max_exp
                    mant[j] = max_mant
                    
                # Store scale like this for easy multiplication
                # Actual implementation only stores scale once per block
                scale[j] = shared_exp 
                    
        float_out = sign * mant * (2 ** (scale + exp))
        
        # Reconstruct the tensor
        tensor_reconstructed = float_out.view(tensor_shape)
        
        return tensor_reconstructed
                        