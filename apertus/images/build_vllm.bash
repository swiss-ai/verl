#!/bin/bash

eval $(ssh-agent -s)
podman build --ssh default=$SSH_AUTH_SOCK -t apertus:vllm -f apertus_vllm.dockerfile .
