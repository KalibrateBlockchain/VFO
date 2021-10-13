FROM ubuntu:20.04

WORKDIR /app

COPY . ./

COPY ./docker_utils/requirements.txt ./

RUN apt-get update -y  && apt-get upgrade -y

RUN apt-get install python3 -y

RUN apt install python3-pip -y

RUN pip3 install -r requirements.txt

RUN apt-get install -y cron

COPY ./docker_utils/cronjob /etc/cron.d/

RUN chmod 0644 /etc/cron.d/cronjob && crontab /etc/cron.d/cronjob

RUN crontab -l

EXPOSE 80


