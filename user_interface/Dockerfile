FROM python:3.9

WORKDIR /code

COPY app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user app $HOME/app

RUN ["chmod", "+x", "/home/user/app/run_app.sh"]

CMD /home/user/app/run_app.sh
