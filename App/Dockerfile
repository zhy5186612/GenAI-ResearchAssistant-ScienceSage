# Use a python image
FROM python:3.11.0 AS build

RUN apt-get update -qq && apt-get install ffmpeg -y

# add new user
RUN adduser --disabled-password ssuser
USER ssuser
ENV PATH="/home/ssuser/.local/bin:${PATH}"

#
WORKDIR /home/ssuser
EXPOSE 8011

#### Build imge and Run the container as kguser
RUN mkdir .streamlit
COPY --chown=ssuser:ssuser mb-qs-pp-cuda07.na.pg.com.key .
COPY --chown=ssuser:ssuser mb-qs-pp-cuda07.na.pg.com.crt .
COPY --chown=ssuser:ssuser .streamlit/config.toml .streamlit
COPY --chown=ssuser:ssuser requirements.txt .
COPY --chown=ssuser:ssuser utils.py .
COPY --chown=ssuser:ssuser prompts.py .
COPY --chown=ssuser:ssuser sciSageApp.py .
COPY --chown=ssuser:ssuser scienceSage.png .

# install the required packages
RUN pip install pip --upgrade
RUN pip install --user -r requirements.txt
RUN pip3 install git+https://github.com/openai/CLIP.git --user
RUN pip install git+https://github.com/openai/whisper.git
#
CMD streamlit run sciSageApp.py
