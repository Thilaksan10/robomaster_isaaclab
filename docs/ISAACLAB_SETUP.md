# Isaac Lab Instructions

These are instructions for setting up a remote docker container to run isaac lab, while also being able to connect.

## Start
For setting up the environment, the isaac lab docker container on a remote machine is used. We mount the workspace to the docker-compose file, as well as a bash history. To write the bash history, we need to add history -a to PROMPT_COMMAND env variable.

Mounting docker-compose:

```yaml
- type: bind
  source: ../../ws_mobile_manipulation
  target: ${DOCKER_ISAACLAB_PATH}/../ws_mobile_manipulation

- type: bind
  source: .bash/.bash_history
  target: /root/.bash_history
```

In Dockerfile:

```bash
echo "export PROMPT_COMMAND='history -a'" >> ${HOME}/.bashrc
```

 ```bash
 
 USER root
# Add User 
ARG USERNAME=user # your user on the remote machine
ARG USER_UID=1004 # you can get these with id -u
ARG USER_GID=1005 # you can get these with id -g

# add local user
RUN groupadd -g $USER_GID $USERNAME && \
    useradd -u $USER_UID -g $USER_GID $USERNAME -m -s /bin/bash && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "root:root" | chpasswd

 # adjust aliasing isaaclab.sh and python for convenience, delete the other aliasing block instead of this one
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /home/$USERNAME/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> /home/$USERNAME/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /home/$USERNAME/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /home/$USERNAME/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> /home/$USERNAME/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> /home/$USERNAME/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> /home/$USERNAME/.bashrc && \
    echo "export TZ=$(date +%Z)" >> /home/$USERNAME/.bashrc && \
    echo "export PROMPT_COMMAND='history -a'" >> /home/$USERNAME/.bashrc && \
    chown -R $USERNAME:$USERNAME /workspace && \
    chown -R $USERNAME:$USERNAME /isaac-sim

```


The following aliases are use for the local host and remote respectively:

local machine (omni_stream):

```bash 
   alias omni_stream='${HOME}/.local/share/ov/pkg/kit_remote-103.1.1/kit-remote -w 1920 -h 1080 -s IP_REMOTE_HOST'
```

remote machine:

```bash
   alias lab='/PATH/TO/IsaacLab/docker/container.py'
```

Now isaac lab can be controlled on the remote machine with

```bash 
lab start/stop
```

and VSCode can be connected to the remote docker container. Now a the mounted repository can be fitted with a train.py (e.g. standalone/workflow/rl_games/train.py). To register tasks an you can create a new task folder in ws/tasks. This has to be fitted with an __init__.py, e.g.,

```python
import gymnasium as gym

from .Robomsater import RobomasterEnv, RobomasterEnvCfg
import cfg_train

gym.register(
    id="Robomaster",
    entry_point=RobomasterEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RobomasterEnvCfg,
        "rl_games_cfg_entry_point": f"{cfg_train.__name__}:rl_games_ppo_cfg.yaml",

    }
)

```
if you created a new task ws/tasks/franka_mm.py.
