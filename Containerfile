FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python39 python39-pip python39-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

WORKDIR /meps_ml

RUN git clone https://github.com/fmidev/meps_ml_correction.git

RUN update-alternatives --set python3 /usr/bin/python3.9 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
