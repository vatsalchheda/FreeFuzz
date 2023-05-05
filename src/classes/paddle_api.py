import paddle
from classes.argument import *
from classes.api import *
from classes.database import PaddleDatabase


def generate_code_for_complex_tensor(shape, var_name, t):
    if t == paddle.complex64:
        x = paddle.float32
    else:
        x = paddle.float64
    code = f"real = paddle.rand({shape}, {x})\nimag = paddle.rand({shape}, {x})\n{var_name}_tensor = paddle.complex(real, imag)\n"
    return code


def generate_code_for_int8_tensor(shape, var_name):
    code = f"int_tensor = paddle.randint(low=-128, high=127, shape={shape}, dtype='int32')\nint8_tensor = int_tensor.astype('int8')\n{var_name}_tensor = int8_tensor\n"
    return code

def generate_code_for_uint8_tensor(shape, var_name):
    code = f"int_tensor = paddle.randint(low=0, high=255, shape={shape}, dtype='int32')\nuint8_tensor = int_tensor.astype('uint8')\n{var_name}_tensor = uint8_tensor\n"
    return code


def generate_code_for_int16_tensor(shape, var_name):
    code = f"int_tensor = paddle.randint(low=-32768, high=32767, shape={shape}, dtype='int32')\nint16_tensor = int_tensor.astype('int16')\n{var_name}_tensor = int16_tensor\n"
    return code


def generate_code_for_f16_tensor(shape, var_name):
    code = f"float_tensor = paddle.rand({shape}, 'float32')\nf16_tensor = float_tensor.astype('float16')\n{var_name}_tensor = f16_tensor\n"
    return code


class PaddleArgument(Argument):
    _support_types = [ArgType.PADDLE_DTYPE, ArgType.PADDLE_TENSOR, ArgType.PADDLE_OBJECT]

    _dtypes = [
        paddle.bfloat16, paddle.bool, paddle.complex128, paddle.complex64,
        paddle.uint8, paddle.int8, paddle.int16, paddle.int32, paddle.int64,
        paddle.float32, paddle.float64, paddle.float16
    ]

    def __init__(self,
                 value,
                 type: ArgType,
                 shape=None,
                 dtype=None,
                 max_value=1,
                 min_value=0):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value
    def to_code(self, var_name, low_precision=False, is_cuda=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision,
                                              is_cuda)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.PADDLE_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)
            suffix = ""
            if is_cuda:
                suffix = ".cuda()"
            if dtype in [paddle.float32, paddle.float64]: # float
                code = f"{var_name}_tensor = paddle.rand({self.shape}, dtype={dtype})\n"
            elif dtype in [paddle.int32, paddle.int64]: # int
                code = f"{var_name}_tensor = paddle.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            elif dtype == paddle.bool: # bool
                code = f"{var_name}_tensor = paddle.randint(0,2,{self.shape})\n"
            elif dtype in [paddle.complex64, paddle.complex128]:
                code = generate_code_for_complex_tensor(self.shape, var_name, dtype)
            elif dtype == paddle.int8:
                code = generate_code_for_int8_tensor(self.shape, var_name)
            elif dtype == paddle.int16:
                code = generate_code_for_int16_tensor(self.shape, var_name)
            elif dtype == paddle.uint8:
                code = generate_code_for_uint8_tensor(self.shape, var_name)
            elif dtype == paddle.float16:
                code = generate_code_for_f16_tensor(self.shape, var_name)
            else:
                code = f"{var_name}_tensor = paddle.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            code += f"{var_name} = {var_name}_tensor.clone(){suffix}\n"
            return code
        elif self.type == ArgType.PADDLE_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.PADDLE_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.PADDLE_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().astype({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.PADDLE_OBJECT:
            pass
        elif self.type == ArgType.PADDLE_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.PADDLE_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert (0)

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    PaddleArgument(2, ArgType.INT),
                    PaddleArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.PADDLE_TENSOR:
                self.shape = [2, 2]
                self.dtype = paddle.float32
            elif new_type == ArgType.PADDLE_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.PADDLE_OBJECT:
                pass
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.PADDLE_TENSOR:
            new_size = list(self.shape)
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type == ArgType.PADDLE_OBJECT:
            pass
        elif self.type == ArgType.PADDLE_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert (0)

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Paddle argument from the signature"""
        if signature == "paddleTensor":
            return PaddleArgument(None,
                                 ArgType.PADDLE_TENSOR,
                                 shape=[2, 2],
                                 dtype=paddle.float32)
        if signature == "paddledtype":
            return PaddleArgument(choice(PaddleArgument._dtypes),
                                 ArgType.PADDLE_DTYPE)
        if isinstance(signature, str) and signature == "paddledevice":
            value = paddle.set_device("cpu")
            return PaddleArgument(value, ArgType.PADDLE_OBJECT)
        # if isinstance(signature, str) and signature == "torch.strided":
        #     return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("paddle."):
            value = eval(signature)
            if isinstance(value, paddle.dtype):
                return PaddleArgument(value, ArgType.PADDLE_DTYPE)
            print(signature)
            assert (0)
        if isinstance(signature, bool):
            return PaddleArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return PaddleArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return PaddleArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return PaddleArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(PaddleArgument.generate_arg_from_signature(elem))
            return PaddleArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(PaddleArgument.generate_arg_from_signature(elem))
            return PaddleArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature['dtype']
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("paddle."):
                    dtype = f"paddle.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = PaddleArgument.random_tensor_value(dtype)
                return PaddleArgument(None,
                                     ArgType.PADDLE_TENSOR,
                                     shape,
                                     dtype=dtype,
                                     max_value=max_value,
                                     min_value=min_value)
            else:
                return PaddleArgument(None,
                                     ArgType.PADDLE_TENSOR,
                                     shape=[2, 2],
                                     dtype=paddle.float32)
        return PaddleArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [paddle.int16, paddle.int32, paddle.int64]:
            return paddle.int8
        elif dtype in [paddle.float16, paddle.float32, paddle.float64]:
            return paddle.float32
        elif dtype in [paddle.complex128]:
            return paddle.complex64
        return dtype

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == paddle.bool:
            max_value = 2
            min_value = 0
        elif dtype == paddle.uint8:
            max_value = 1 << randint(0, 9)
            min_value = 0
        elif dtype == paddle.int8:
            max_value = 1 << randint(0, 8)
            min_value = -1 << randint(0, 8)
        elif dtype == paddle.int16:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        else:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        return max_value, min_value

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, paddle.Tensor):
            return ArgType.PADDLE_TENSOR
        elif isinstance(x, paddle.dtype):
            return ArgType.PADDLE_DTYPE
        else:
            return ArgType.PADDLE_OBJECT
# TODO: create paddle_api class
class PaddleAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        if record == None:
            record = PaddleDatabase.get_rand_record(self.api)
        self.args = self.generate_args_from_record(record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = PaddleDatabase.select_rand_over_db(
                    self.api, arg_name)
                if success:
                    new_arg = PaddleArgument.generate_arg_from_signature(
                        new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code(self,
                prefix="arg",
                res="res",
                is_cuda=False,
                use_try=False,
                error_res=None,
                low_precision=False) -> str:
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_code(arg_name,
                                low_precision=low_precision,
                                is_cuda=is_cuda)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if is_cuda:
                code += f"{prefix}_class = {self.api}({arg_str}).cuda()\n"
            else:
                code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_code(
                    arg_name, low_precision=low_precision, is_cuda=is_cuda)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           low_precision)

    @staticmethod
    def invocation_code(res, error_res, res_code, use_try, low_precision):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res == None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f"except Exception as e:\n{error_res} = \"ERROR:\"+str(e)\n"
            res_code = temp_code

        if low_precision:
            code += "start = time.time()\n"
            code += res_code
            code += f"{res} = time.time() - start\n"
        else:
            code += res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = PaddleArgument.generate_arg_from_signature(
                    record[key])
        return args

    def to_diff_code(self,
                     oracle: OracleType,
                     prefix="arg",
                     res="res",
                     *,
                     error_res=None,
                     use_try=False) -> str:
        """Generate code for the oracle"""
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_diff_code(arg_name, oracle)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if oracle == OracleType.CUDA:
                code = f"{prefix}_class = {prefix}_class.cuda()\n"
            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_diff_code(arg_name, oracle)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           oracle == OracleType.PRECISION)