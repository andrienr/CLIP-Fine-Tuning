#!/bin/sh

python3 manage.py collectstatic
python3 -m gunicorn marble_catalogue.wsgi:application --reload --workers=1 --timeout 0 -b 0.0.0.0:7860
# python3 manage.py runserver 0.0.0.0:7860