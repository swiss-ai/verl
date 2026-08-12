# syntax=docker/dockerfile:1
FROM docker.io/verlai/verl:vllm023.aarch64.dev1

COPY xielu.patch /tmp/xielu.patch

RUN git clone https://github.com/rubber-duck-debug/xielu.git && \
    cd xielu && \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0" CMAKE_CUDA_ARCHITECTURES="9.0" pip install . --no-deps --no-build-isolation  && \
    pip install flash-attn-4[cu13]==4.0.0b19 && \
    pip install instanttensor && \
    cd /vllm && \
    git apply /tmp/xielu.patch

RUN cd / && \ 
    pip install nvidia-modelopt && \
    git clone https://github.com/wqwqazwsxedc/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout apertus && \
    pip install -e . && \
    git clone https://github.com/wqwqazwsxedc/Megatron-Bridge.git && \
    cd Megatron-Bridge && \
    git checkout apertus && \
    pip install --no-deps -e . 

RUN pip install cupy-cuda13x \
    math-verify \ 
    latex2sympy2-extended \
    nltk \
    langdetect \
    immutabledict \
    emoji \
    syllapy

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:swiss-ai/tool-gym.git && \
    cd tool-gym && \
    git checkout fix-verl-packaging-paths && \
    pip install -e . && \
    git clone https://github.com/wqwqazwsxedc/r-gym.git && \
    cd r-gym && \
    git checkout translate && \
    pip install -e .
    
