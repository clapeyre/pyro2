FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1 WARP_CACHE_PATH=/tmp/warp-cache
WORKDIR /work
COPY tools/warp/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
COPY . /opt/pyro2
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYRO_HYDRO=4.5.2.dev0 pip install --no-build-isolation --no-deps -e /opt/pyro2
ENV PYTHONPATH=/work
WORKDIR /opt/pyro2
CMD ["python", "-m", "pytest", "-q", "pyro"]
