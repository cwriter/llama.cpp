# llama.cpp for SYCL

- [Background](#background)
- [Recommended Release](#recommended-release)
- [News](#news)
- [OS](#os)
- [Hardware](#hardware)
- [Performance Reference](#performance-reference)
- [Docker](#docker)
- [Linux](#linux)
- [Windows](#windows-1)
- [Environment Variable](#environment-variable)
- [Design Rule](#design-rule)
- [Known Issue](#known-issues)
- [Q&A](#qa)
- [TODO](#todo)

## Background

**SYCL** is a high-level parallel programming model designed to improve developers productivity writing code across various hardware accelerators such as CPUs, GPUs, and FPGAs. It is a single-source language designed for heterogeneous computing and based on standard C++17.

**oneAPI** is an open ecosystem and a standard-based specification, supporting multiple architectures including but not limited to Intel CPUs, GPUs and FPGAs. The key components of the oneAPI ecosystem include:

- **DPCPP** *(Data Parallel C++)*: The primary oneAPI SYCL implementation, which includes the icpx/icx Compilers.
- **oneAPI Libraries**: A set of highly optimized libraries targeting multiple domains *(e.g. Intel oneMKL, oneMath and oneDNN)*.
- **oneAPI LevelZero**: A high performance low level interface for fine-grained control over Intel iGPUs and dGPUs.

### Llama.cpp + SYCL

The llama.cpp SYCL backend is primarily designed for **Intel GPUs**.
SYCL cross-platform capabilities enable support for other vendor GPUs as well.

## Recommended Release

### Windows

The following releases are verified and recommended:

|Commit ID|Tag|Release|Verified  Platform| Update date|
|-|-|-|-|-|
|24e86cae7219b0f3ede1d5abdf5bf3ad515cccb8|b5377 |[llama-b5377-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b5377/llama-b5377-bin-win-sycl-x64.zip) |Arc B580/Linux/oneAPI 2025.1<br>LNL Arc GPU/Windows 11/oneAPI 2025.1.1|2025-05-15|
|3bcd40b3c593d14261fb2abfabad3c0fb5b9e318|b4040 |[llama-b4040-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b4040/llama-b4040-bin-win-sycl-x64.zip) |Arc A770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1| 2024-11-19|
|fb76ec31a9914b7761c1727303ab30380fd4f05c|b3038 |[llama-b3038-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b3038/llama-b3038-bin-win-sycl-x64.zip) |Arc A770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1||

### Ubuntu 24.04

The release packages for Ubuntu 24.04 x64 (FP32/FP16) only include the binary files of the llama.cpp SYCL backend. They require the target machine to have pre-installed Intel GPU drivers and oneAPI packages that are the same version as the build package. To get the version and installation info, refer to [.github/workflows/release.yml#L713](../../.github/workflows/release.yml#L713): ubuntu-24-sycl -> Download & Install oneAPI.

It is recommended to use them with [Intel Docker](https://hub.docker.com/r/intel/deep-learning-essentials).

The packages for FP32 and FP16 would have different accuracy and performance on LLMs. Please choose it according to the test result.

## News

- 2026.04-05
  - Optimize mul_mat by reorder feature for data type: Q4_K, Q5_K, Q6_K, Q8_0.
  - Fused MoE.
  - Upgrate CI and built package for oneAPI 2025.3.3, support Ubuntu 24.04 built package.

- 2026.03
  - Support Flash-Attention: less memory usage, performance impact depends on LLM.

- 2026.02
  - Remove support for Nvidia & AMD GPU, because the oneAPI plugin for Nvidia & AMD GPU is unavailable: download/installation channels are out of work. User can't build up the software for Nvidia & AMD GPU.

- 2025.11
  - Support malloc memory on device more than 4GB.

- 2025.2
  - Optimize MUL_MAT Q4_0 on Intel GPU for all dGPUs and built-in GPUs since MTL. Increase the performance of LLM (llama-2-7b.Q4_0.gguf) 21%-87% on Intel GPUs (MTL, ARL-H, Arc, Flex, PVC).
    |GPU|Base tokens/s|Increased tokens/s|Percent|
    |-|-|-|-|
    |PVC 1550|39|73|+87%|
    |Flex 170|39|50|+28%|
    |Arc A770|42|55|+30%|
    |MTL|13|16|+23%|
    |ARL-H|14|17|+21%|

- 2024.11
  - Use syclcompat to improve the performance on some platforms. This requires to use oneAPI 2025.0 or newer.

- 2024.8
  - Use oneDNN as the default GEMM library, improve the compatibility for new Intel GPUs.

- 2024.5
  - Performance is increased: 34 -> 37 tokens/s of llama-2-7b.Q4_0 on Arc A770.
  - Arch Linux is verified successfully.

- 2024.4
  - Support data types: GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M.

- 2024.3
  - Release binary files of Windows.
  - A blog is published: **Run LLM on all Intel GPUs Using llama.cpp**: [intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llm-on-all-gpus-using-llama-cpp-artical.html) or [medium.com](https://medium.com/@jianyu_neo/run-llm-on-all-intel-gpus-using-llama-cpp-fd2e2dcbd9bd).
  - New base line is ready: [tag b2437](https://github.com/ggml-org/llama.cpp/tree/b2437).
  - Support multiple cards: **--split-mode**: [none|layer]; not support [row], it's on developing.
  - Support to assign main GPU by **--main-gpu**, replace $GGML_SYCL_DEVICE.
  - Support detecting all GPUs with level-zero and same top **Max compute units**.
  - Support OPs
    - hardsigmoid
    - hardswish
    - pool2d

- 2024.1
  - Create SYCL backend for Intel GPU.
  - Support Windows build

## OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Linux   | Support | Ubuntu 22.04, Fedora Silverblue 39, Arch Linux |
| Windows | Support | Windows 11                                     |


## Hardware

### Intel GPU

SYCL backend supports Intel GPU Family:

- Intel Data Center Max Series
- Intel Flex Series, Arc Series
- Intel Built-in Arc GPU
- Intel iGPU in Core CPU (11th Generation Core CPU and newer, refer to [oneAPI supported GPU](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html#inpage-nav-1-1)).

On older Intel GPUs, you may try [OpenCL](/docs/backend/OPENCL.md) although the performance is not optimal, and some GPUs may not support OpenCL nor have any GPGPU capabilities.

#### Verified devices

| Intel GPU                     | Status  | Verified Model                        |
|-------------------------------|---------|---------------------------------------|
| Intel Data Center Max Series  | Support | Max 1550, 1100                        |
| Intel Data Center Flex Series | Support | Flex 170                              |
| Intel Arc A-Series            | Support | Arc A770, Arc A730M, Arc A750         |
| Intel Arc B-Series            | Support | Arc B580                              |
| Intel built-in Arc GPU        | Support | built-in Arc GPU in Meteor Lake, Arrow Lake, Lunar Lake |
| Intel iGPU                    | Support | iGPU in 13700k, 13400, i5-1250P, i7-1260P, i7-1165G7  |

*Notes:*

- **Memory**
  - The device memory is a limitation when running a large model. The loaded model size, *`llm_load_tensors: buffer_size`*, is displayed in the log when running `./bin/llama-completion`.
  - Please make sure the GPU shared memory from the host is large enough to account for the model's size. For e.g. the *llama-2-7b.Q4_0* requires at least 8.0GB for integrated GPU and 4.0GB for discrete GPU.

- **Execution Unit (EU)**
  - If the iGPU has less than 80 EUs, the inference speed will likely be too slow for practical use.

### Other Vendor GPU

NA

## Performance Reference


To get the supported LLMs, GPUs, and performance reference, please check [Performance of llama.cpp on Intel GPU with SYCL backend](https://github.com/ggml-org/llama.cpp/discussions/23313).

You could update your test result in it directly.

## Docker

Please refer to [Docker with SYCL](../docker.md#docker-with-sycl) for details.

## Quick Development WOW

This chapter is for quick development & try with SYCL backend on Intel GPU.

You need to install following sofeware before development:
   - Intel GPU driver
   - oneAPI package
   - other development tools.

Please refer to [Linux](#linux) or [Windows](#windows-1) for above installation and resolve the trouble in usage. There are the detailed guide.

- Linux

```
## build from source code
./examples/sycl/build.sh

## run CONV_2D_DW unit test cases
./build/bin/test-backend-ops -b SYCL0 -o CONV_2D_DW

## run all unit test cases
./build/bin/test-backend-ops -b SYCL0

## run with LLM on the first GPU
./examples/sycl/test.sh -mg 0 -m xxxx.gguf

## run service with LLM on the first GPU
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
./examples/sycl/start-svr.sh -m xxxx.gguf

## update the docs/ops.md for new/update OPs
./examples/sycl/update-ops-doc.sh
```

- Windows

```
## build from source code
examples\sycl\win-build-sycl.bat

## run CONV_2D_DW unit test cases
build\bin\test-backend-ops.exe -b SYCL0 -o CONV_2D_DW

## run all unit test cases
build\bin\test-backend-ops.exe -b SYCL0

## run LLM on the first GPU
examples\sycl\win-test.bat -mg 0 -m xxxx.gguf

## run service with LLM on the first GPU
set ONEAPI_DEVICE_SELECTOR="level_zero:0"
examples\sycl\win-start-svr.bat -m xxxx.gguf

## update the docs/ops.md for new/update OPs
examples\sycl\win-update-ops-doc.bat
```


## Linux

### I. Setup Environment

1. **Install GPU drivers**

  - **Intel GPU**

Intel data center GPUs drivers installation guide and download page can be found here: [Get Intel dGPU Drivers](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps).

*Note*: for client GPUs *(iGPU & Arc A-Series)*, please refer to the [client iGPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html).

Once installed, add the user(s) to the `video` and `render` groups.

```sh
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```

*Note*: logout/re-login for the changes to take effect.

Verify installation through `clinfo`:

```sh
sudo apt install clinfo
sudo clinfo -l
```

Sample output:

```sh
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics

Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Iris(R) Xe Graphics [0x9a49]
```

2. **Install Intel® oneAPI Base toolkit**

SYCL backend depends on:
  - Intel® oneAPI DPC++/C++ compiler/running-time.
  - Intel® oneAPI DPC++/C++ library (oneDPL).
  - Intel® oneAPI Deep Neural Network Library (oneDNN).
  - Intel® oneAPI Math Kernel Library (oneMKL).

- **For Intel GPU**

All above are included in both **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** packages.

It's recommended to install **Intel® Deep Learning Essentials** which only provides the necessary libraries with less size.

The **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Linux, and preferably keep the default installation values unchanged, notably the installation path *(`/opt/intel/oneapi` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

Upon a successful installation, SYCL is enabled for the available Intel devices, along with relevant libraries such as oneAPI oneDNN for Intel GPUs.

|Verified release|
|-|
|2025.3.3 |
|2025.2.1|
|2025.1|
|2024.1|

3. **Verify installation and environment**

In order to check the available SYCL devices on the machine, please use the `sycl-ls` command.
```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

- **Intel GPU**

When targeting an intel GPU, the user should expect one or more devices among the available SYCL devices. Please make sure that at least one GPU is present via `sycl-ls`, for instance `[level_zero:gpu]` in the sample output below:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.3.29735+27]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) UHD Graphics 730 12.2.0 [1.3.29735+27]
[opencl:cpu][opencl:0] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i5-13400 OpenCL 3.0 (Build 0) [2025.20.8.0.06_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.39.31294]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) UHD Graphics 730 OpenCL 3.0 NEO  [24.39.31294]
```

### II. Build llama.cpp

#### Intel GPU

```sh
# Uses FP32, consider using FP16 for better performance in most cases
./examples/sycl/build.sh
```

or

```sh
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP16 (recommended for better performance in most cases)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

# Option 2: Use FP32
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# build all binary
cmake --build build --config Release -j -v
```

It is possible to come across some precision issues when running tests that stem from using faster
instructions, which can be circumvented by setting the environment variable `SYCL_PROGRAM_COMPILE_OPTIONS`
as `-cl-fp32-correctly-rounded-divide-sqrt`

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Obtaining and quantizing models*](../../README.md#obtaining-and-quantizing-models) guide for model preparation, or download an already quantized model like [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf?download=true) or [Meta-Llama-3-8B-Instruct-Q4_0.gguf](https://huggingface.co/aptha/Meta-Llama-3-8B-Instruct-Q4_0-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf).

##### Check device

1. Enable oneAPI running environment

```sh
source /opt/intel/oneapi/setvars.sh
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```sh
./build/bin/llama-ls-sycl-device
```

This command will only display the selected backend that is supported by SYCL. The default backend is level_zero. For example, in a system with 2 *Intel GPU* it would look like the following:
```
found 2 SYCL devices:

|  |                  |                                             |Compute   |Max compute|Max work|Max sub|               |
|ID|       Device Type|                                         Name|capability|units      |group   |group  |Global mem size|
|--|------------------|---------------------------------------------|----------|-----------|--------|-------|---------------|
| 0|[level_zero:gpu:0]|               Intel(R) Arc(TM) A770 Graphics|       1.3|        512|    1024|     32|    16225243136|
| 1|[level_zero:gpu:1]|                    Intel(R) UHD Graphics 770|       1.3|         32|     512|     32|    53651849216|
```

#### Choose level-zero devices

|Chosen Device ID|Setting|
|-|-|
|0|`export ONEAPI_DEVICE_SELECTOR="level_zero:0"` or no action|
|1|`export ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|0 & 1|`export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"`|

#### Execute

Choose one of following methods to run.

1. Script

- Use device 0:

```sh
./examples/sycl/test.sh -mg 0
```
- Use multiple devices:

```sh
./examples/sycl/test.sh
```

- Run llama-server:

```sh
./examples/sycl/start-svr.sh -m PATH/MODEL_FILE
```

2. Command line
Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user. Default device id is 0.
- Multiple devices: Automatically choose the devices with the same backend.

In two device selection modes, the default SYCL backend is level_zero, you can choose other backend supported by SYCL by setting environment variable ONEAPI_DEVICE_SELECTOR.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |
| Multiple devices | --split-mode tensor (tensor parallelism) |

`--split-mode tensor` (tensor parallelism) shards each layer across the selected
GPUs. It requires flash attention, which is auto-enabled when `--flash-attn` is
left at its default `auto`, so `--split-mode tensor` works out of the box.
Passing `--flash-attn off` together with `--split-mode tensor` is rejected at
context creation. The default `f16` KV cache is recommended. Tensor parallelism
is currently optimized for 2 GPUs; other device counts fall back to a generic
all-reduce.

Examples:

- Use device 0:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-completion -no-cnv -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 99 -sm none -mg 0 --load-mode auto
```

- Use multiple devices:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-completion -no-cnv -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 99 -sm layer --load-mode auto
```

*Notes:*

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```
Or
```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```

User can use the device management in [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), like parameter `--device SYCL0,SYCL1` to assign one or more devices.

## Windows

### Install GPU driver

Intel GPU drivers instructions guide and download page can be found here: [Get Intel GPU Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html).

### Option 1: download the binary package directly

Download the binary package for Windows from: https://github.com/ggml-org/llama.cpp/releases.

Extract the package to local folder, run the llama tools directly. Refer to [Run the inference](#iii-run-the-inference-1).

Note, the package includes the SYCL running time and all depended dll files, no need to install oneAPI package and activte them.

### Option 2: build locally from the source code.

#### I. Setup environment

1. Install Visual Studio

If you already have a recent version of Microsoft Visual Studio, you can skip this step. Otherwise, please refer to the official download page for [Microsoft Visual Studio](https://visualstudio.microsoft.com/).

2. Install Intel® oneAPI Base toolkit

SYCL backend depends on:
  - Intel® oneAPI DPC++/C++ compiler/running-time.
  - Intel® oneAPI DPC++/C++ library (oneDPL).
  - Intel® oneAPI Deep Neural Network Library (oneDNN).
  - Intel® oneAPI Math Kernel Library (oneMKL).

All above are included in both **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** packages.

It's recommended to install **Intel® Deep Learning Essentials** which only provides the necessary libraries with less size.

The **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Windows, and preferably keep the default installation values unchanged, notably the installation path *(`C:\Program Files (x86)\Intel\oneAPI` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

b. Enable oneAPI running environment:

- Type "oneAPI" in the search bar, then open the `Intel oneAPI command prompt for Intel 64 for Visual Studio 2022` App.

- On the command prompt, enable the runtime environment with the following:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

- if you are using Powershell, enable the runtime environment with the following:

```
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```

c. Verify installation

In the oneAPI command line, run the following to print the available SYCL devices:

```
sycl-ls.exe
```

There should be one or more *level-zero* GPU devices displayed as **[ext_oneapi_level_zero:gpu]**. Below is example of such output detecting an *Intel Iris Xe* GPU as a Level-zero SYCL device:

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [31.0.101.5186]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.28044]
```

3. Install build tools

a. Download & install cmake for Windows: https://cmake.org/download/ (CMake can also be installed from Visual Studio Installer)
b. The new Visual Studio will install Ninja as default. (If not, please install it manually: https://ninja-build.org/)


#### II. Build llama.cpp

You could download the release package for Windows directly, which including binary files and depended oneAPI dll files.

Choose one of following methods to build from source code.

##### Option 1: Script

```sh
# Uses FP32, consider using FP16 for better performance in most cases
.\examples\sycl\win-build-sycl.bat
```

##### Option 2: CMake

On the oneAPI command line window, step into the llama.cpp main directory and run the following:

```
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

# Option 1: Use FP16 (recommended for better performance in most cases)
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON

# Option 2: Or FP32
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j
```

Or, use CMake presets to build:

```sh
cmake -DGGML_SYCL_F16=ON --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-completion

cmake --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-completion

cmake --preset x64-windows-sycl-debug
cmake --build build-x64-windows-sycl-debug -j --target llama-completion
```

##### Option 3: Visual Studio

You have two options to use Visual Studio to build llama.cpp:
- As CMake Project using CMake presets.
- Creating a Visual Studio solution to handle the project.

**Note**:

All following commands are executed in PowerShell.

###### - Open as a CMake Project

You can use Visual Studio to open the `llama.cpp` folder directly as a CMake project. Before compiling, select one of the SYCL CMake presets:

- `x64-windows-sycl-release`

- `x64-windows-sycl-debug`

*Notes:*
- For a minimal experimental setup, you can build only the inference executable using:

    ```Powershell
    cmake --build build --config Release -j --target llama-completion
    ```

###### - Generating a Visual Studio Solution

You can use Visual Studio solution to build and work on llama.cpp on Windows. You need to convert the CMake Project into a `.sln` file.

If you want to use the Intel C++ Compiler for the entire `llama.cpp` project, run the following command:

```Powershell
cmake -B build -G "Visual Studio 17 2022" -T "Intel C++ Compiler 2025" -A x64 -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release
```

If you prefer to use the Intel C++ Compiler only for `ggml-sycl`, ensure that `ggml` and its backend libraries are built as shared libraries ( i.e. `-DBUILD_SHARED_LIBRARIES=ON`, this is default behaviour):

```Powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
      -DSYCL_INCLUDE_DIR="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" \
      -DSYCL_LIBRARY_DIR="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
```

If successful the build files have been written to: *path/to/llama.cpp/build*
Open the project file **build/llama.cpp.sln** with Visual Studio.

Once the Visual Studio solution is created, follow these steps:

1. Open the solution in Visual Studio.

2. Right-click on `ggml-sycl` and select **Properties**.

3. In the left column, expand **C/C++** and select **DPC++**.

4. In the right panel, find **Enable SYCL Offload** and set it to `Yes`.

5. Apply the changes and save.


*Navigation Path:*

```
Properties -> C/C++ -> DPC++ -> Enable SYCL Offload (Yes)
```

Now, you can build `llama.cpp` with the SYCL backend as a Visual Studio project.
To do it from menu: `Build -> Build Solution`.
Once it is completed, final results will be in **build/Release/bin**

*Additional Note*

- You can avoid specifying `SYCL_INCLUDE_DIR` and `SYCL_LIBRARY_DIR` in the CMake command by setting the environment variables:

    - `SYCL_INCLUDE_DIR_HINT`

    - `SYCL_LIBRARY_DIR_HINT`

- Above instruction has been tested with Visual Studio 17 Community edition and oneAPI 2025.0. We expect them to work also with future version if the instructions are adapted accordingly.

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Obtaining and quantizing models*](../../README.md#obtaining-and-quantizing-models) guide for model preparation, or download an already quantized model like [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) or [Meta-Llama-3-8B-Instruct-Q4_0.gguf](https://huggingface.co/aptha/Meta-Llama-3-8B-Instruct-Q4_0-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf).

##### Check device

1. Enable oneAPI running environment

On the oneAPI command line window, run the following and step into the llama.cpp directory:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```
build\bin\llama-ls-sycl-device.exe
```

This command will only display the selected backend that is supported by SYCL. The default backend is level_zero. For example, in a system with 2 *Intel GPU* it would look like the following:
```
found 2 SYCL devices:
|  |                  |                                             |Compute   |Max compute|Max work|Max sub|               |
|ID|       Device Type|                                         Name|capability|units      |group   |group  |Global mem size|
|--|------------------|---------------------------------------------|----------|-----------|--------|-------|---------------|
| 0|[level_zero:gpu:0]|               Intel(R) Arc(TM) A770 Graphics|       1.3|        512|    1024|     32|    16225243136|
| 1|[level_zero:gpu:1]|                    Intel(R) UHD Graphics 770|       1.3|         32|     512|     32|    53651849216|

```

##### Choose level-zero devices

|Chosen Device ID|Setting|
|-|-|
|0|Default option. You may also want to `set ONEAPI_DEVICE_SELECTOR="level_zero:0"`|
|1|`set ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|0 & 1|`set ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"` or `set ONEAPI_DEVICE_SELECTOR="level_zero:*"`|

##### Execute

Choose one of following methods to run.

1. Script

- Run test:

```
examples\sycl\win-test.bat
```

- Run llama-server:

```
examples\sycl\win-start-svr.bat -m PATH\MODEL_FILE
```

2. Command line

Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user. Default device id is 0.
- Multiple devices: Automatically choose the devices with the same backend.

In two device selection modes, the default SYCL backend is level_zero, you can choose other backend supported by SYCL by setting environment variable ONEAPI_DEVICE_SELECTOR.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |
| Multiple devices | --split-mode tensor (tensor parallelism) |

`--split-mode tensor` (tensor parallelism) shards each layer across the selected
GPUs. It requires flash attention, which is auto-enabled when `--flash-attn` is
left at its default `auto`, so `--split-mode tensor` works out of the box.
Passing `--flash-attn off` together with `--split-mode tensor` is rejected at
context creation. The default `f16` KV cache is recommended. Tensor parallelism
is currently optimized for 2 GPUs; other device counts fall back to a generic
all-reduce.

Examples:

- Use device 0:

```
build\bin\llama-completion.exe -no-cnv -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 99 -sm none -mg 0 --load-mode auto
```

- Use multiple devices:

```
build\bin\llama-completion.exe -no-cnv -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 99 -sm layer --load-mode auto
```


Note:

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```

Or

```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```

User can use the device management in [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), like parameter `--device SYCL0,SYCL1` to assign one or more devices.

## Environment Variable

### Build

| Name               | Value                                 | Function                                    |
|--------------------|---------------------------------------|---------------------------------------------|
| GGML_SYCL          | ON (mandatory)                        | Enable build with SYCL code path.           |
| GGML_SYCL_TARGET   | INTEL *(default)*                     | Set the SYCL target device type.            |
| GGML_SYCL_DEVICE_ARCH | Optional                           | Set the SYCL device architecture. Setting the device architecture can improve the performance. See the table [--offload-arch](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OffloadDesign.md#--offload-arch) for a list of valid architectures. |
| GGML_SYCL_F16      | OFF *(default)* \|ON *(optional)*     | Enable FP16 build with SYCL code path. (1.) |
| GGML_SYCL_GRAPH    | ON *(default)* \|OFF *(Optional)*     | Enable build with [SYCL Graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc). |
| GGML_SYCL_DNN      | ON *(default)* \|OFF *(Optional)*     | Enable build with oneDNN.                   |
| GGML_SYCL_HOST_MEM_FALLBACK | ON *(default)* \|OFF *(Optional)* | Allow host memory fallback when device memory is full during quantized weight reorder. Enables inference to continue at reduced speed (reading over PCIe) instead of failing. Requires Linux kernel 6.8+. |
| GGML_SYCL_SUPPORT_LEVEL_ZERO_API | ON *(default)* \|OFF *(Optional)* | Support to use Level Zero API for device memory allocation. Requires Level Zero headers/library at build time and Intel GPU driver (Level Zero runtime) at run time. Reduces system RAM usage during multi-GPU inference. SYCL backend always runs on Level Zero running time even if it's set as OFF (The SYCL api will be usage for memory allocation).|
| CMAKE_C_COMPILER   | `icx` *(Linux)*, `icx/cl` *(Windows)* | Set `icx` compiler for SYCL code path.      |
| CMAKE_CXX_COMPILER | `icpx` *(Linux)*, `icx` *(Windows)*   | Set `icpx/icx` compiler for SYCL code path. |

1. FP32 or FP16 have different performance impact to LLM. Recommended to test them for better prompt processing performance on your models. You need to rebuild the code after change `GGML_SYCL_F16=OFF/ON`.

### Runtime

| Name              | Value            | Function                                                                                                                  |
|-------------------|------------------|---------------------------------------------------------------------------------------------------------------------------|
| GGML_SYCL_DEBUG   | 0 (default) or 1 | Enable log function: GGML_SYCL_DEBUG() for common debug. |
| GGML_SYCL_DEV_DEBUG   | 0 (default) or 1 | Enable log function: GGML_SYCL_DEV_DEBUG() for developmental purposes by replacing GGML_SYCL_DEBUG() in special codes. Restore to GGML_SYCL_DEBUG() before committing code.|
| GGML_SYCL_DEV2DEV_MEMCPY | 0 (default), 1, 2 | Choose the method of dev2dev memory copy.<br>Value: <br>*  0: SYCL API (default), only support dGPUs.<br>* 1: L0 API -- Better performance, only support dGPUs, found to lead to abnormal crash in some case. <br>* 2: Host Forward -- Most stable method for all cases (including iGPU + dGPU*N), but with lower performance (-2% to -5%).<br>SYCL & L0 API are easy to be impacted by Intel GPU driver issue. When you meet the garbled output or crash issues in multiple GPUs case, try with this debug flag to work around or check the issue.|
| GGML_SYCL_ENABLE_FLASH_ATTN | 1 (default) or 0| Enable Flash-Attention. It can reduce memory usage. The performance impact depends on the LLM.|
| GGML_SYCL_ENABLE_OPT | 0 or 1 (default)| Enable optimize features for Intel GPUs. (Recommended to 0 for Intel devices older than Gen 10) |
| GGML_SYCL_ENABLE_GRAPH | 0 (default) or 1 | Enable running computations through SYCL Graphs feature. Disabled by default because SYCL Graph is still on development, performance depends on device. |
| GGML_SYCL_ENABLE_HOST_PINNED_MEM | 0 or 1 (default) | Enable host pinned memory to speed up copy data from host to device. When disable it, host memory will common malloc() on CPU. Disable it when use `--load-model mlock`.|
| GGML_SYCL_HOST_PINNED_MEM_2G | 0 (default) or 1 | Limit the max memory allocation to be no more than 2GB when enable host pinned memory. USM allocations above 2 GiB take the relaxed/large-allocation path, which serializes H2D copies with compute and prevents copy/compute overlap. It will impact the startup time. Need more test. Depend on `GGML_SYCL_ENABLE_HOST_PINNED_MEM=1`.|
| GGML_SYCL_GET_MEM_API | 0 (default) or 1  | Set to get memory info (free, total) by Level Zero or SYCL API:<br>0 - Level Zero API: support more GPUs, only run on Level Zero running time. When there is an error, fallback to call SYCL API. Depend on GGML_SYCL_SUPPORT_LEVEL_ZERO_API.<br>1 - SYCL API: legacy, support more running time, it can't get the free size of some GPUs (like Arc770). In such case, return the free size as value of total size.|
| GGML_SYCL_USE_LEVEL_ZERO_API | 1 (default) or 0 | Use Level Zero API for device memory allocation instead of SYCL. Reduces system RAM usage on Intel dGPUs by avoiding DMA-buf/TTM host memory staging. Requires GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON at build time. SYCL backend always runs on Level Zero running time even if it's set as OFF (The SYCL api will be usage for memory allocation).|
| GGML_SYCL_ENABLE_DNN | 0 or 1 (default)| Enable running computations through oneDNN and always use oneMKL. |
| GGML_SYCL_FA_ONEDNN | 1 (default) or 0 | Enable the oneDNN fused SDPA (flash-attention) path on supported GPUs. Set to 0 to always use the native SYCL flash-attention kernel. |
| GGML_SYCL_FA_ONEDNN_MAX_KV | 0 (default, disabled) or positive integer | By default (0), all sequences are handled by the oneDNN fused SDPA path, regardless of KV length; a positive value caps that length, past which sequences fall back to the native kernel. If GPU driver watchdog resets (DEVICE_LOST) occur during long-context inference, set this near the context depth where they start, e.g. 24576. |
| GGML_SYCL_ENABLE_VMM | 0 or 1 (default) | Enable the virtual-memory device pool. |
| GGML_SYCL_ENABLE_MKL_FA | 1 (default), 2 or 0 | Enable oneMKL GEMM flash attention for XMX-accelerated prompt processing with quantized KV cache. Automatically activates during prefill (prompt processing) when all conditions are met: (1) flash-attn enabled (`-fa` or `--flash-attn on`), (2) KV cache quantized (`--cache-type-k q8_0 --cache-type-v q8_0` or other `*_0/*_1` types), (3) batch size ≥ 1024 (`--batch-size 1024`), (4) prompt length ≥ 1024 tokens. Set to 0 to force the TILE kernel for A/B testing. Set to 2 to keep oneMKL flash attention but normalize the output one head at a time (the older, slower kernel), for A/B testing. Example minimum command: `llama-cli -m model.gguf -fa -ngl 99 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 1024 -p "your prompt"` |
| GGML_SYCL_MKL_FA_DEBUG | 0 (default) or 1 | Enable per-call diagnostic logging for MKL flash attention: GEMM/softmax timings, interleaved-head detection, and buffer memory usage. |
| GGML_SYCL_MEMTRACE | 0 (default), 1, 2 | Enable record and output memory allocation diagnostics. Requires `-lv 4`. <br>0 -  Disable<br>1 - Basic memory info, including current and peak allocations, as well allocations from other sources, around 50 lines per model load.<br>2 - More verbose, logging around 900 specific allocations and deallocations. |
| GGML_SYCL_MEMTRACE_STEP | 64 (default) or positive integer | With GGML_SYCL_MEMTRACE=1, the minimum growth in memory usage to trigger another log record. |
| GGML_SYCL_MKL_FA_DIAG | 0 (default) or 1 | Enable output fingerprinting for MKL flash attention. Dumps the first 64 float output values for the first 6 FA calls with n_kv ≥ 1024, labeled with kernel type (MKL/TILE/VEC) for cross-kernel comparison. |
| GGML_SYCL_ENABLE_FUSION | 0 or 1 (default) | Enable fused-kernel dispatch in graph compute. Unsupported types and layouts fall back to the standalone op kernels. See `ggml_sycl_can_fuse()`. |
| GGML_SYCL_REORDER_TYPES | bitmask, all but Q8_0 by default | Which quant types may have their MoE expert weights reordered into a per-expert structure-of-arrays layout at first use, so a row's quants are contiguous instead of interleaved with their scales. Bits: 1 = IQ3_S, 2 = IQ4_NL, 4 = Q8_0. Q8_0 is off by default because its MoE mat-vec is a small share of decode and its quants are already contiguous within a block, so the reorder measures flat. Set to 0 to keep the canonical block layout for every type. |
| GGML_SYCL_MOE_REORDER | -1 (default), 0, or 1 | Control GPU-side expert-major route ordering for quantized MoE matrix operations. The default enables it for at least 64 routed rows. Set to 0 for the direct route-major kernel or 1 to force expert-major ordering. |
| GGML_SYCL_MOE_XMX | 0 or 1 (default) | Enable the Q8_0 XMX consumer for expert-major MoE matrix operations when the device supports signed-int8 joint matrices and the average expert bucket has at least 8 routes. Set to 0 to use the SLM-tiled DP4A consumer. |
| GGML_SYCL_ENABLE_ESIMD | 0 or 1 (default)| Enable ESIMD kernels when available. |
| GGML_SYCL_SPARSE_FA<br>GGML_SYCL_SPARSE_FA_DEBUG<br>GGML_SYCL_SPARSE_FA_MARGIN| 0 (default) or 1<br>0 (default) or 1<br>[0,..] default:256| Enable Sparse Flash-attention.<br>Enable to debug for Sparse Flash-attention.<br>Set the margin value for Sparse Flash-attention.|
| GGML_SYCL_KQ_MASK_BITS | 0 (default) or 1 | Pack the causal mask to one bit per KV cell in device memory. The tensor keeps its f16 type and shape, so only this backend sees the difference. **Not expected to give a speedup**, and off by default: the mask is ~0.4% of what a flash-attention kernel reads (2 bytes per cell against 2*head_size for K and V), so shrinking it 16x is invisible. Correct, but currently of no benefit. |
| GGML_SYCL_XMX_GATHER_TYPES | decimal bitmask, all bits set (default) | Select which quantized weight formats may take the XMX dequant-GEMM paths, where the weights are dequantized inside the GEMM (gathered straight into the XMX tiles) instead of being written out to f16 and read back. This covers the grouped `MUL_MAT_ID` path used by MoE models, and the plain `MUL_MAT` path when built with `GGML_SYCL_F16=ON` (the plain path sits inside that build's f16 branch). Both compute in f16 on the XMX units regardless of `GGML_SYCL_F16`, so enabling them for `MUL_MAT_ID` trades some precision for speed relative to the per-expert library GEMM they replace. Mainly affects prompt processing; token generation is unaffected. One bit per format, so a format can be enabled or benchmarked on its own:<br>* 1: IQ4_NL<br>* 2: IQ3_S<br>* 4: IQ4_XS<br>* 8: IQ3_XXS<br>* 16: IQ2_XXS<br>* 32: IQ2_XS<br>* 64: IQ2_S<br>* 128: IQ1_S<br>* 256: IQ1_M<br>* 512: Q8_0<br>* 1024: Q4_K<br>* 2048: Q5_K<br>* 4096: Q6_K<br>The four non-IQ formats also have an A stage for the reorder (SoA) layout, so they keep the path when `GGML_SYCL_ENABLE_OPT` has rewritten the weights; Q4_K, Q5_K and Q6_K are reordered unconditionally on `MUL_MAT_ID`, so for those the SoA stage is the only one that ever runs there. Set to 0 to disable the paths entirely and fall back to the library GEMM, which is the baseline to compare against. A format is only taken when the shape also fits (the weights must cover whole blocks, and the tile is only used while N is narrow), so setting a bit does not force the path. Formats outside this list are never affected by this variable. |
| GGML_SYCL_FAST_AND_SLOPPY | 0 (default) or 1 | Allow XMX dequant-GEMM paths that are faster but give up accuracy in edge cases the library GEMM handles. Currently this gates `q4_K` on the gather paths (bit 1024 of `GGML_SYCL_XMX_GATHER_TYPES` has no effect without it). The fused A stage carries the dequantized weight as f16, so a weight whose magnitude exceeds the f16 range (65504) saturates to infinity and its product becomes NaN. Trained weights are many orders of magnitude below that, so there is no known real-world effect; the case that does hit it is synthetic, `test-backend-ops -o MUL_MAT_ID` with `type_a=q4_K,n_mats=128,n_used=4,m=4096,k=2048,amax=100000`, which fails with this enabled and passes with it off. `q5_K` and `q6_K` share the same f16 A stage but no failing case is known for them, so they are not gated. |
| GGML_SYCL_MMID_SCHED | decimal bitmask, 0 (default) | Tuning bits for the `MUL_MAT_ID` grouped XMX dequant-GEMM: how its tile schedule is built, and how its tiles are shaped. All off by default, so the host-built schedule stays the correctness oracle.<br>* 1: build the tile schedule on the device, so the host never drains the queue to read the routing back.<br>* 2: skip the slice-width heuristic, which the device arm cannot ask because the routing is not on the host.<br>* 4: read the device schedule back and check it against the host one.<br>* 8: run both arms and compare the output bitwise.<br>* 16: keep the host schedule, but pad it to the device arm's bound, to separate the cost of the wider launch from the cost of building the schedule on the device.<br>* 32: pack B at the tile's own first row instead of at a tile-aligned slot, so the packed-B buffer holds one column per routed row rather than one per tile slot.<br>* 64: give a work-group 32 rows of the output instead of 16. A work-group streams the whole packed-B tile of its column group, so half as many row-groups is half the packed-B read traffic; the kernel is L3-bound on that re-read. The cost is register pressure (8 accumulator tiles instead of 4), 4 KiB more shared local memory per work-group, and half the work-groups, so measure it rather than assume it.<br>* 128: build the 32-row kernel of bit 64 with 256 registers per thread instead of 128. Only that kernel; the 16-row one always keeps the default. 8 accumulator tiles is about 64 of 128 registers before anything else, so this is the lever to pull if the profiler reports register spilling. 256 registers halve the threads a vector engine can hold, which is free only while shared local memory is already the tighter limit on how many work-groups fit. Has no effect unless bit 64 is also set.<br>* 256: give a work-group 64 rows instead of 16, one step further than bit 64. This kernel is always built with 256 registers per thread, whatever bit 128 says: 16 accumulator tiles are about 128 of 128 registers on their own, so 128-register mode would spill for certain. Shared local memory goes to 24 KiB per work-group (16 for the staged weights, 8 for the K-split buffer, which stays 16 rows wide). Wins over bit 64 when both are set. Bit 64 moved L3 read traffic by far less than the halved packed-B re-read predicted, so most of that traffic is not the packed-B re-read and this bit is unlikely to repay its registers; measure it.<br>* 512: stage the raw quantized weight bytes through shared local memory. A lane owns one row, so the lane-to-lane stride of every weight field is a whole row of blocks: each of the format's streams touches one cache line per lane to deliver a handful of bytes, and the same line then serves the 8 k steps of its superblock but is evicted by the ~32 KiB of packed B that streams past in between. With this bit the lanes of a sub-group instead copy one stored block of every row of the M tile into shared local memory cooperatively -- contiguous runs, so one line per run rather than one line per lane -- and then decode their own row out of it, which also makes the block resident for all of its k steps. The quantized bytes are staged, not the dequantized halves: one row costs one block (110 B for iq3_s) instead of 8 k steps of 32 f16, so the cost is `FG_KSPLIT * rows * sizeof(block)`, 6.9 KiB at 16 rows and 13.8 KiB at 32 for iq3_s, on top of the 12 and 16 KiB the work-group already asks for. Both the reordered (SoA) and the canonical layout are staged, and every weight format the grouped GEMM decodes is covered. Not offered on the 64-row tile of bit 256, which already asks for 24 KiB, and not on the plain non-grouped fused GEMM, which is not the kernel the weight gather dominates. Shared local memory per work-group is what limits how many of them a vector engine holds, so this trades occupancy for L3 read traffic; measure it.<br>* 1024: software pipeline the weight stage. A k step today stages its weights into shared local memory, waits on a barrier, then loads them as matrix tiles: no weight load ever overlaps a multiply-accumulate, which is what this kernel looks like in a profile -- the execution units are idle about 92 percent of the time while neither the arithmetic nor the bandwidth is the limit. With this bit the staged tile is doubled and the stage of k step n+1 goes into the second copy while the multiply-accumulates of step n read the first, so the barrier before a step publishes the copy the previous step staged instead of the copy its own matrix loads are about to read. One barrier per k step instead of two. Shared local memory per work-group goes from 12 to 16 KiB at 16 rows and from 16 to 24 KiB at 32 rows: 16 KiB is what bit 64 asks for today and that measured faster, 24 KiB is what bit 256 asks for and that measured slower, so measure the 16-row tile first. Not offered on the 64-row tile of bit 256, which would go to 40 KiB, and not combined with bit 512, which adds back the barrier this bit removes; when either is selected this bit is ignored.<br>* 2048: hold one stored weight block in registers across all of its k steps. A k step is 32 weight values, but a superblock format stores 256 of them together, so 8 consecutive k steps read the same stored block and today each of them re-issues all of the format's global reads for it -- about 32 KiB of packed B streams past in between, which evicts the lines. With this bit the k loop walks stored blocks instead of k steps: the block's fields are read once into registers and each of its 8 sub-blocks is decoded from there. This is the same temporal reuse bit 512 buys, without the shared local memory round trip and without its barrier per block, so the two are never combined and bit 512 wins when both are set; bit 1024 is ignored as well. Bit 128 chooses the register count here as it does everywhere else: at 256 registers the kernel measured no spill at all, but 256 registers halve the threads a vector engine holds and that cost more than the removed re-reads saved, and at 128 registers an iq3_s superblock is 110 bytes a lane on top of the accumulators and it spills. Reordered (SoA) weights only -- the canonical layout keeps the single-shot weight stages -- and not on the 64-row tile of bit 256, where a lane owns four rows and so four blocks at once. Register pressure is the whole risk here: the block is 110 bytes for iq3_s, 144 for q4_K, 176 for q5_K and 212 for q6_K, doubled on the 32-row tile of bit 64 where a lane owns two rows, so a format may still spill at 256 registers; measure it per format. Formats with 32 values per block (iq4_nl, q8_0) store one k step per block, so the bit changes nothing for them.<br>* 4096: hold only the cheap streams of a stored weight block in registers, and keep re-reading the wide ones per k step. Bit 2048 removes every re-read of a block by holding all of it, which for iq3_s is 110 bytes, about 28 registers a row on top of the 52 the accumulators and the matrix operands already take. That does not fit in 128 registers, because the sub-block loop is fully unrolled -- deliberately, since a runtime sub-block index would put the block in scratch memory -- so the whole block stays live across eight decode and multiply-accumulate sequences. But a stream is not re-read in proportion to its width: every stream of the block is re-read once per k step whatever its size, so the 6 bytes of iq3_s metadata remove as many re-reads as its 64 bytes of quants do. This bit caches the streams that cost at most 4 registers a block and leaves the rest streaming. For iq3_s that is the metadata (the superblock scale and the packed sub-block scales) and qh, which is one byte per k step: 4 registers a row, and half of the block's 32 global reads removed. For q4_K, q5_K and q6_K it is the scales and the superblock scale factors, 5 registers a row. qs, signs, and the wide qh of q5_K and q6_K keep being read per k step, from the same addresses the uncached path reads them. Unlike bit 2048 this does not force 256 registers per thread -- fitting in 128 is the point of it -- but bit 128 still selects 256 if you want to measure both. Reordered (SoA) weights only, not on the 64-row tile of bit 256, and never combined with bits 512, 1024 or 2048; bit 2048 wins when both are set, being the wider variant of the same idea. Formats with 32 values per block (iq4_nl, q8_0) store one k step per block, so there is no re-read to remove and they keep the plain kernel.<br>* 32768: lay the staged weight tile out as whole 8x16 matrix tiles instead of row major. A sub-group multiplies its 16 rows as two 8x16 matrix tiles a k step, and each tile is loaded out of shared local memory with the tile's row stride, which is 32 halves: the 16 halves a tile row needs are half of that stride, so a tile is not one contiguous run and the load costs one scalar 32-byte message per row. That is 32 messages a k step, more than half of every memory message the k loop issues, on data the same thread wrote a few instructions earlier. With this bit the image holds each 8x16 tile as 256 contiguous bytes, so a tile loads in one message and the k loop issues 4 of them instead of 32. The decode writes its 32 halves into registers first and stores them as the two runs the tiled layout puts them in, so no weight format's decode changes. Shared local memory does not move: the same halves are written and read, only at different addresses, and the barriers that publish the tile are the same. Offered on all three tile widths and with bits 512 and 1024, but not with bits 2048 or 4096, which hold a stored block in registers already and spill at 128 registers; when either is set this bit is ignored.<br>* 65536: hold the iq3_s lookup table in shared local memory, one copy per work-group. The iq3_s weight decode reads 8 table entries per k step per lane, at indices taken from the quantized bytes, so the lanes diverge and the compiler emits 8 stateless 64-bit-addressed gathers plus 8 64-bit address builds per k step, in a loop body of about 640 instructions. That decode is the larger half of this kernel -- with the multiply-accumulates skipped it is 58 percent of the time, and its execution units are still 89 percent idle, so it is waiting on memory rather than computing. The table is 512 dwords, 2 KiB, small enough for a work-group to hold all of it: the 64 work items copy it once at kernel entry, one barrier publishes it, and every gather after that is 32-bit addressed and never leaves the subslice. Checked offline with ocloc on the real kernel: the 8 `load.ugm.d32.a64` in the k loop become 8 `load.slm.d32.a32`, the 8 64-bit address builds go with them, and the copy is one 14-instruction loop and one work-group barrier before the k loop. The copy is once per kernel, not per k step and not per stored block. The cost is 2 KiB of shared local memory per work-group, which is what limits how many of them a vector engine holds here: the 16-row tile goes from 12 to 14 KiB, so floor(128/12) = 10 resident work-groups become floor(128/14) = 9; the 32-row tile of bit 64 goes from 16 to 18 KiB (8 to 7) and the 64-row tile of bit 256 from 24 to 26 KiB (5 to 4). Bit 512 is the warning: it added about 7 KiB and a barrier per stored block and measured 31 percent slower, so this is a trade, not a free win. Both the reordered (SoA) and the canonical weight stage read the staged copy. iq3_s only -- it is the one format here whose table is both read per k step and small enough to stage -- and not combined with bits 2048 or 4096, whose decode keeps reading the table where it lives today. |
| ZES_ENABLE_SYSMAN | 0 (default) or 1 | Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory.<br>Recommended to use when --split-mode = layer |
| UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS | 0 (default) or 1 | Allow SYCL/Unified Runtime Level Zero device allocations larger than 4 GiB. llama.cpp's direct Level Zero allocation path requests the relaxed maximum-size limit itself when GGML_SYCL_ENABLE_LEVEL_ZERO=1. |
| GGML_SYCL_USM_SYSTEM | decimal bitmask, 0 (default) | Enable experimental support for [USM system allocations](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html#system-allocations). This requires an Intel Xe2+ GPU such as BMG or newer and is supported on Linux only, with CONFIG_DRM_XE_GPUSVM enabled; the device must report `usm_system_allocations`, and the bits do nothing if it does not.<br>* 1: back a SYCL buffer of 4 GiB or more with the system allocator instead of device memory. Needs enough host memory for the model weights and caches. This was the whole meaning of the variable when it was a boolean, so `GGML_SYCL_USM_SYSTEM=1` keeps its old behaviour.<br>* 2: **currently disabled and ignored** - gather a host-mapped weight on the device. A weight kept out of VRAM with `-ot <name>=CPU` is mapped from the model file and normally makes the scheduler run its `GET_ROWS` on the CPU, then stage the dense result to the device once per token. With this bit the backend claims that buffer type, so the gather runs on the device, reads the rows it needs straight out of host memory, and only the row indices cross the bus. Aimed at a per-layer embedding table far larger than VRAM, where the rows read are a small fraction of the result. The backend claims such a weight for `GET_ROWS` only: any other op that reads one is left where it was, because it would stream the whole weight across the bus every token. That also means those ops can no longer be offloaded by copy during prompt processing, so do not set this bit together with `-ot` overrides that keep expert weights on the host. |

## Compile-time Flags

Pass these via `CXXFLAGS` or add a one-off `#define` to enable a flag on the spot.

| Name            | Function                                                                         |
|-----------------|----------------------------------------------------------------------------------|
| DEBUG_SYCL_POOL | Enable device memory pool logging on teardown. Useful for profiling allocations. |
| DEBUG_SYCL_MALLOC | Enable verbose per-call logging of device pool alloc/free operations. |
| GGML_SYCL_SUPPORT_VMM | Support to building with VMM code. Default is Yes. |

## Design Rule

- Open to all contributors.

- All code change should be useful to user:
    - Fix bug.
    - Add new function.
    - Improve the performance/usage.
    - Make code be easy to maintain.
    - ...

- Don't accept the codes of following cases:
    - Break legacy function.
    - Reduce the performance of legacy case in default.
    - Not completed work/the functionality cannot be demonstrated.

- Encourage to use environment variable to control features to be opened/closed.
    - User can evaluate the feature without rebuild the code.
    - Recommend the best features to user by setting them be opened as default.

- Design the code based on the published official releases of oneAPI packages: compiler, library, driver, OS kernel.

- Developers need to maintain the code they submit.

## Known Issues

- `Split-mode:[row]` is not supported.

- Missed the AOT (Ahead-of-Time) in building.
  - Good: Builds quickly, smaller size of binary file.
  - Bad: The startup is slow (JIT) in first time, but subsequent performance is unaffected.

## Q&A

- Error:  `error while loading shared libraries: libsycl.so: cannot open shared object file: No such file or directory`.

  - Potential cause: Unavailable oneAPI installation or not set ENV variables.
  - Solution: Install *oneAPI base toolkit* and enable its ENV through: `source /opt/intel/oneapi/setvars.sh`.

- General compiler error:

  - Remove **build** folder or try a clean-build.

- I can **not** see `[ext_oneapi_level_zero:gpu]` after installing the GPU driver on Linux.

  Please double-check with `sudo sycl-ls`.

  If it's present in the list, please add video/render group to your user then **logout/login** or restart your system:

  ```
  sudo usermod -aG render $USER
  sudo usermod -aG video $USER
  ```
  Otherwise, please double-check the GPU driver installation steps.

- Can I report Ollama issue on Intel GPU to llama.cpp SYCL backend?

  No. We can't support Ollama issue directly, because we aren't familiar with Ollama.

  Suggest reproducing on llama.cpp and report similar issue to llama.cpp. We will support it.

  It's same for other projects including llama.cpp SYCL backend.

- `Native API failed. Native API returns: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`, `ggml_backend_sycl_buffer_type_alloc_buffer: can't allocate 3503030272 Bytes of memory on device`, or `failed to allocate SYCL0 buffer`

  You are running out of Device Memory.

  |Reason|Solution|
  |-|-|
  | The default context is too big. It leads to excessive memory usage.|Set `-c 8192` or a smaller value.|
  | The model is too big and requires more memory than what is available.|Choose a smaller model or change to a smaller quantization, like Q5 -> Q4;<br>Alternatively, use more than one device to load model.|

- `ggml_backend_sycl_buffer_type_alloc_buffer: can't allocate 5000000000 Bytes of memory on device`

  With the default `GGML_SYCL_ENABLE_LEVEL_ZERO=1`, llama.cpp requests Level Zero's relaxed maximum-size allocation limit directly. If Level Zero support is disabled at build time or runtime and the allocation goes through SYCL/Unified Runtime instead, enable support for allocations larger than 4 GiB by:
  ```
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    set UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
  ```

- When I set `SYCL_CACHE_PERSISTENT=1` in running time, I meet crash.

  `SYCL_CACHE_PERSISTENT=1` is not recommended by llama.cpp SYCL backend.
  When cache is enabled, SYCL runtime will try to cache and reuse JIT-compiled binaries.

  We find some AI will tell user this cmd to speed up SYCL backend. It only speeds up the startup to skip the JIT process, instead of running speed.

  It will bring negative impact when the SYCL binary file is changed frequently in your running environment. The new & old codes mix will lead to crash.

  Compare to the benefit, it has brought more failed cases.
  If you are not familiar with the SYCL compiler principle of JIT and AOT, please don't use it.

  To restore, you need to remove the local cache: `~/.cache/libsycl_cache/` and execute `unset SYCL_CACHE_PERSISTENT` in running time.

- How to use iGPU and dGPU in same time?

  1. Detect the devices in your running time.
  ```
  source /opt/intel/oneapi/setvars.sh
  ./build/bin/llama-server --list-devices

  or
  ./build/bin/llama-cli --list-devices
  ./build/bin/llama-bench --list-devices
  ./build/bin/llama-completion --list-devices

  Available devices:
    SYCL0: Intel(R) Arc(TM) A770 Graphics (15473 MiB, 15473 MiB free)
    SYCL1: Intel(R) UHD Graphics 770 (59675 MiB, 44986 MiB free)
  ```

  The dGPU will be in the head of this list and iGPU will be the end.
  If not all GPUs are listed, please check the env var: ONEAPI_DEVICE_SELECTOR and unset it.

  2. Set the iGPU and dGPU

  Set the iGPU and dGPU by `./build/bin/llama-server --device SYCL0,SYCL1,SYCLxxx`.


### **GitHub contribution**:
Please add the `[SYCL]` prefix/tag in issues/PRs titles to help the SYCL contributors to check/address them without delay.

## TODO

- Review ZES_ENABLE_SYSMAN: https://github.com/intel/compute-runtime/blob/master/programmers-guide/SYSMAN.md#support-and-limitations
