FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python39 python39-pip python39-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

RUN git clone https://github.com/fmidev/meps_ml_correction.git

WORKDIR /meps_ml_correction

ADD https://lake.fmi.fi/ml-models/meps-ml-correction/meps_lsm.grib /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/meps_topography.grib /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/all_stations_windspeed.csv /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_windspeed_20231214.npz /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_windspeed_20231214.json /meps_ml_correction

RUN chmod 644 meps_lsm.grib && \
    chmod 644 meps_topography.grib && \
    chmod 644 all_stations_windspeed.csv && \
    chmod 644 quantiles_windspeed_20231214.npz && \
    chmod 644 xgb_windspeed_20231214.json && \
    update-alternatives --set python3 /usr/bin/python3.9 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
