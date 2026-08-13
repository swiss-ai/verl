FROM apertus_rl:vllm

WORKDIR /rl_packages

RUN git clone https://github.com/swiss-ai/tool-gym.git && \
    cd tool-gym && \
    git checkout fix-verl-packaging-paths && \
    pip install -e . && \
    git clone https://github.com/wqwqazwsxedc/r-gym.git && \
    cd r-gym && \
    git checkout translate && \
    pip install -e .
    