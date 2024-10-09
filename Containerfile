FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm && \
    dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python39 python39-pip python39-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf && \
    git clone https://github.com/fmidev/meps_ml_correction.git

WORKDIR /meps_ml_correction

ENV WS_TAG=20231214
ENV WG_TAG=20240304
ENV TA_TAG=20241009
ENV TD_TAG=20241009
ENV TMAX_TAG=20241009
ENV TMIN_TAG=20241009

ADD https://lake.fmi.fi/ml-models/meps-ml-correction/meps_lsm.grib /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/meps_topography.grib /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/all_stations_windspeed.csv /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/all_stations_windgust.csv /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/all_stations_temperature.csv /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/all_stations_dewpoint.csv /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_windspeed_$WS_TAG.json /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_windgust_$WG_TAG.json /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_temperature_$TA_TAG.json /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_dewpoint_$TD_TAG.json /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_t_max_$TMAX_TAG.json /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/xgb_t_min_$TMIN_TAG.json /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_windspeed_$WS_TAG.npz /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_windgust_$WG_TAG.npz /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_temperature_$TA_TAG.npz /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_dewpoint_$TD_TAG.npz /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_t_max_$TMAX_TAG.npz /meps_ml_correction
ADD https://lake.fmi.fi/ml-models/meps-ml-correction/quantiles_t_min_$TMIN_TAG.npz /meps_ml_correction

RUN chmod 644 meps_lsm.grib && \
    chmod 644 meps_topography.grib && \
    chmod 644 all_stations_windspeed.csv && \
    chmod 644 all_stations_windgust.csv && \
    chmod 644 all_stations_temperature.csv && \
    chmod 644 all_stations_dewpoint.csv && \
    chmod 644 quantiles_windspeed_$WS_TAG.npz && \
    chmod 644 xgb_windspeed_$WS_TAG.json && \
    chmod 644 quantiles_windgust_$WG_TAG.npz && \
    chmod 644 xgb_windgust_$WG_TAG.json && \
    chmod 644 quantiles_temperature_$TA_TAG.npz && \
    chmod 644 xgb_temperature_$TA_TAG.json && \
    chmod 644 quantiles_dewpoint_$TD_TAG.npz && \
    chmod 644 xgb_dewpoint_$TD_TAG.json && \
    chmod 644 quantiles_t_max_$TMAX_TAG.npz && \
    chmod 644 xgb_t_max_$TMAX_TAG.json && \
    chmod 644 quantiles_t_min_$TMIN_TAG.npz && \
    chmod 644 xgb_t_min_$TMIN_TAG.json && \
    update-alternatives --set python3 /usr/bin/python3.9 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
