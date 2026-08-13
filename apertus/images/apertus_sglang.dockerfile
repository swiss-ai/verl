FROM docker.io/verlai/verl:sgl0512.aarch64.dev1

RUN git clone https://github.com/rubber-duck-debug/xielu.git && \
    cd xielu && \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0" CMAKE_CUDA_ARCHITECTURES="9.0" pip install . --no-deps --no-build-isolation  && \
    pip install flash-attn-4[cu13]==4.0.0b19 && \
    pip install instanttensor

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
    pip install --ignore-installed -e . && \
    git clone https://github.com/wqwqazwsxedc/r-gym.git && \
    cd r-gym && \
    git checkout translate && \
    pip install -e .

RUN pip install --break-system-packages nvidia-cutlass-dsl==4.5.2 \
    nvidia-cutlass-dsl-libs-cu13==4.5.2 
COPY xielu_sglang.patch /tmp/xielu.patch
RUN cd /sgl-workspace/sglang && \
    git apply /tmp/xielu.patch
