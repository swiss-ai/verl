#!/bin/bash

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_ed25519
podman build --ssh default=$SSH_AUTH_SOCK -t apertus:sglang -f apertus_sglang.dockerfile .
