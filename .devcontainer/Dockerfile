FROM isaacgym:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    xvfb ffmpeg
    
RUN pip install --upgrade pip

WORKDIR /opt
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
RUN cd IsaacGymEnvs &&\
    git checkout dee7c56765e14f6f4344c4d2e91d7a9eb3bfa619 &&\
    pip install -e .

USER gymuser

RUN pip install gym==0.23.1 wandb